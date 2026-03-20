import os

from google.genai import Client

from agent.configuration import Configuration
from agent.prompts import get_current_date, web_searcher_instructions
from agent.search_providers.base import BaseSearchProvider
from agent.utils import get_citations, insert_citation_markers, resolve_urls


class GeminiGoogleSearchProvider(BaseSearchProvider):
    """Search provider backed by Gemini's google_search tool."""

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._client = Client(api_key=api_key)

    def web_research(
        self,
        search_query: str,
        idx: int,
        *,
        model: str,
        logger,
        sep: str,
    ) -> dict:
        formatted_prompt = web_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=search_query,
        )
        logger.info(
            f"\n{sep}\n[web_research] REQUEST\n  model : {model}\n"
            f"  query : {search_query}\n  tools : google_search\n"
            f"  prompt:\n{formatted_prompt}\n{sep}"
        )

        response = self._client.models.generate_content(
            model=model,
            contents=formatted_prompt,
            config={
                "tools": [{"google_search": {}}],
                "temperature": 0,
            },
        )
        grounding_chunks = (
            response.candidates[0].grounding_metadata.grounding_chunks
            if response.candidates and response.candidates[0].grounding_metadata
            else []
        )
        source_urls = [c.web.uri for c in grounding_chunks if hasattr(c, "web")]
        logger.info(
            f"\n{sep}\n[web_research] RESPONSE\n  model  : {model}\n"
            f"  query  : {search_query}\n  sources: {source_urls}\n"
            f"  text   :\n{response.text}\n{sep}"
        )

        resolved_urls = resolve_urls(grounding_chunks, idx)
        citations = get_citations(response, resolved_urls)
        modified_text = insert_citation_markers(response.text, citations)
        sources_gathered = [
            item for citation in citations for item in citation["segments"]
        ]

        return {
            "sources_gathered": sources_gathered,
            "search_query": search_query,
            "web_research_result": modified_text,
        }
