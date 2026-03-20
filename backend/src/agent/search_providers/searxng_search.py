import json
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from agent.configuration import Configuration
from agent.llm_factory import LLMFactory
from agent.prompts import get_current_date
from agent.search_providers.base import BaseSearchProvider

SEARXNG_SOURCE_PREFIX = "https://search.local/id/"
SOURCE_MARKER_PATTERN = re.compile(r"\[S(\d+)\]")

SEARXNG_SUMMARIZER_INSTRUCTIONS = """You are a research assistant creating a concise report from web search results.

Instructions:
- The current date is {current_date}.
- Use only the sources provided below.
- Do not invent facts or sources.
- When making a factual claim, append one or more source markers like [S1] or [S2][S4].
- Keep the answer readable and well-structured in markdown.
- Preserve uncertainty when the sources disagree or are incomplete.

Research topic:
{research_topic}

Sources:
{sources}
"""


class SearxngSearchProvider(BaseSearchProvider):
    """Search provider backed by a SearxNG JSON API and LLM summarization."""

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        if not config.searxng_base_url:
            raise ValueError("SEARXNG_BASE_URL is not set")

    def web_research(
        self,
        search_query: str,
        idx: int,
        *,
        model: str,
        logger,
        sep: str,
    ) -> dict:
        results = self._search(search_query)
        logger.info(
            f"\n{sep}\n[web_research] REQUEST\n  model : {model}\n"
            f"  query : {search_query}\n  tools : searxng\n"
            f"  result_count : {len(results)}\n{sep}"
        )

        if not results:
            return {
                "sources_gathered": [],
                "search_query": search_query,
                "web_research_result": "No search results were found.",
            }

        source_entries = self._build_source_entries(results, idx)
        formatted_sources = self._format_sources_for_prompt(results)
        prompt = SEARXNG_SUMMARIZER_INSTRUCTIONS.format(
            current_date=get_current_date(),
            research_topic=search_query,
            sources=formatted_sources,
        )

        llm = LLMFactory(self._config).create_chat_model(
            model=model,
            temperature=0,
            max_retries=0,
        )
        response = llm.invoke(prompt)
        response_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        modified_text, used_sources = self._replace_source_markers(
            response_text, source_entries
        )

        logger.info(
            f"\n{sep}\n[web_research] RESPONSE\n  model  : {model}\n"
            f"  query  : {search_query}\n  sources: {[item['value'] for item in used_sources]}\n"
            f"  text   :\n{modified_text}\n{sep}"
        )

        return {
            "sources_gathered": used_sources,
            "search_query": search_query,
            "web_research_result": modified_text,
        }

    def _search(self, query: str) -> list[dict[str, Any]]:
        params = urlencode(
            {
                "q": query,
                "format": "json",
                "language": self._config.searxng_language,
            }
        )
        base_url = self._config.searxng_base_url.rstrip("/")
        with urlopen(f"{base_url}/search?{params}") as response:
            payload = json.load(response)

        results = payload.get("results", [])
        return results[: self._config.searxng_result_limit]

    def _build_source_entries(
        self, results: list[dict[str, Any]], idx: int
    ) -> dict[str, dict[str, str]]:
        source_entries: dict[str, dict[str, str]] = {}
        for result_index, result in enumerate(results, start=1):
            marker = f"S{result_index}"
            source_entries[marker] = {
                "label": self._get_label(result),
                "short_url": f"{SEARXNG_SOURCE_PREFIX}{idx}-{result_index - 1}",
                "value": result.get("url", ""),
            }
        return source_entries

    def _format_sources_for_prompt(self, results: list[dict[str, Any]]) -> str:
        chunks = []
        for result_index, result in enumerate(results, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"[S{result_index}]",
                        f"Title: {result.get('title', '')}",
                        f"URL: {result.get('url', '')}",
                        f"Snippet: {result.get('content', '')}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def _replace_source_markers(
        self,
        text: str,
        source_entries: dict[str, dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        used_sources: list[dict[str, str]] = []
        seen_markers: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            marker = f"S{match.group(1)}"
            source = source_entries.get(marker)
            if not source:
                return match.group(0)
            if marker not in seen_markers:
                used_sources.append(source)
                seen_markers.add(marker)
            return f"[{source['label']}]({source['short_url']})"

        return SOURCE_MARKER_PATTERN.sub(replace, text), used_sources

    def _get_label(self, result: dict[str, Any]) -> str:
        title = result.get("title", "").strip()
        if title:
            return title.split("|")[0].split("-")[0].strip().lower().replace(" ", "-")
        url = result.get("url", "")
        return url.split("/")[2] if "://" in url else "source"
