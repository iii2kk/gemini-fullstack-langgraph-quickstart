"""
Research agent pipeline implemented with pure asyncio (no LangGraph).

Emits the same event structure as the original LangGraph graph:
  - {"generate_query": {"search_query": [...]}}
  - {"web_research": {"sources_gathered": [...]}}
  - {"reflection": {"is_sufficient": bool, ...}}
  - {"finalize_answer": True}
  - {"messages": [...], "sources_gathered": [...]}   <- final result
"""

import os
import re
import asyncio
import logging
import pathlib
from datetime import datetime
from typing import AsyncGenerator, Any, Callable

from dotenv import load_dotenv

from agent.configuration import Configuration
from agent.llm_factory import LLMFactory
from agent.search_providers import create_search_provider
from agent.tools_and_schemas import SearchQueryList, Reflection
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.utils import (
    get_research_topic,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging: console + rotating file (logs/YYYY-MM-DD.log)
# ---------------------------------------------------------------------------

_log_dir = pathlib.Path(__file__).parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

SEP = "=" * 60


# ---------------------------------------------------------------------------
# Rate limiter: 1 API call per RATE_LIMIT_INTERVAL seconds
# ---------------------------------------------------------------------------

RATE_LIMIT_INTERVAL = 0.0  # seconds


class _RateLimiter:
    """Async context manager that enforces a minimum interval between API calls."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = asyncio.Lock()
        self._last_call: float = 0.0

    async def __aenter__(self) -> "_RateLimiter":
        await self._lock.acquire()
        loop = asyncio.get_event_loop()
        wait = self._interval - (loop.time() - self._last_call)
        if wait > 0:
            logger.info(f"[rate_limiter] 次のAPIコールまで {wait:.1f}s 待機中...")
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *_: Any) -> None:
        self._last_call = asyncio.get_event_loop().time()
        self._lock.release()


_rate_limiter = _RateLimiter(RATE_LIMIT_INTERVAL)


def _extract_retry_delay(error_str: str) -> float | None:
    """429 エラー文字列から retryDelay (秒) を取り出す。"""
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", error_str)
    if match:
        return float(match.group(1))
    match = re.search(r"retryDelay[^\d]*(\d+(?:\.\d+)?)s", error_str)
    if match:
        return float(match.group(1))
    return None


async def _rate_limited(fn: Callable, *args: Any, max_attempts: int = 5) -> Any:
    """
    Rate limiter + 429 aware retry.

    - SDK の内部リトライは max_retries=0 で無効化済み。
    - 429 が返ったとき、エラー中の retryDelay を読み取って待機してからリトライ。
    """
    extra_wait = 0.0

    for attempt in range(max_attempts):
        if extra_wait > 0:
            logger.info(
                f"[rate_limiter] 429 のため {extra_wait:.0f}s 追加待機中... "
                f"({attempt}/{max_attempts} 回目)"
            )
            await asyncio.sleep(extra_wait)
            extra_wait = 0.0

        async with _rate_limiter:
            ts = datetime.now().strftime("%H:%M:%S")
            logger.info(f"[rate_limiter] APIコール送信 {ts}  ({attempt + 1}/{max_attempts} 回目)")
            try:
                return await asyncio.to_thread(fn, *args)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    retry_delay = _extract_retry_delay(error_str)
                    if retry_delay is not None and attempt < max_attempts - 1:
                        extra_wait = retry_delay + 5.0
                        logger.warning(
                            f"[rate_limiter] 429 RESOURCE_EXHAUSTED — "
                            f"{extra_wait:.0f}s 後にリトライ ({attempt + 1}/{max_attempts})"
                        )
                        continue  # __aexit__ でロック解放 → ループ先頭で追加待機
                raise

    raise RuntimeError("max_attempts を超えました")

# ---------------------------------------------------------------------------
# Synchronous worker functions (run in thread pool via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _generate_queries_sync(
    messages: list,
    initial_search_query_count: int,
    config: Configuration,
    model: str | None = None,
) -> list:
    llm = LLMFactory(config).create_chat_model(
        model=model or config.query_generator_model,
        temperature=1.0,
        max_retries=0,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)
    formatted_prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(messages),
        number_queries=initial_search_query_count,
    )
    logger.info(f"\n{SEP}\n[generate_query] REQUEST\n  model : {model or config.query_generator_model}\n  prompt:\n{formatted_prompt}\n{SEP}")
    result = structured_llm.invoke(formatted_prompt)
    logger.info(f"\n{SEP}\n[generate_query] RESPONSE\n  queries: {result.query}\n{SEP}")
    return result.query


def _web_research_sync(
    search_query: str,
    idx: int,
    config: Configuration,
    model: str | None = None,
) -> dict:
    if config.search_provider == "gemini_google_search":
        resolved_model = config.query_generator_model
    else:
        resolved_model = model or config.query_generator_model
    provider = create_search_provider(config)
    return provider.web_research(
        search_query,
        idx,
        model=resolved_model,
        logger=logger,
        sep=SEP,
    )


def _reflect_sync(
    messages: list,
    web_research_results: list,
    reasoning_model: str,
    config: Configuration,
) -> dict:
    llm = LLMFactory(config).create_chat_model(
        model=reasoning_model or config.reflection_model,
        temperature=1.0,
        max_retries=0,
    )
    formatted_prompt = reflection_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(messages),
        summaries="\n\n---\n\n".join(web_research_results),
    )
    logger.info(f"\n{SEP}\n[reflection] REQUEST\n  model : {reasoning_model or config.reflection_model}\n  prompt:\n{formatted_prompt}\n{SEP}")
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    logger.info(f"\n{SEP}\n[reflection] RESPONSE\n  is_sufficient  : {result.is_sufficient}\n  knowledge_gap  : {result.knowledge_gap}\n  follow_up      : {result.follow_up_queries}\n{SEP}")
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
    }


def _finalize_answer_sync(
    messages: list,
    web_research_results: list,
    sources_gathered: list,
    reasoning_model: str,
    config: Configuration,
) -> dict:
    llm = LLMFactory(config).create_chat_model(
        model=reasoning_model or config.answer_model,
        temperature=0,
        max_retries=0,
    )
    formatted_prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(messages),
        summaries="\n---\n\n".join(web_research_results),
    )
    logger.info(f"\n{SEP}\n[finalize_answer] REQUEST\n  model : {reasoning_model or config.answer_model}\n  prompt:\n{formatted_prompt}\n{SEP}")
    result = llm.invoke(formatted_prompt)
    logger.info(f"\n{SEP}\n[finalize_answer] RESPONSE\n  text:\n{result.content}\n{SEP}")

    # Replace short URLs with original URLs and collect used sources
    unique_sources = []
    content = result.content
    for source in sources_gathered:
        if source["short_url"] and source["short_url"] in content:
            content = content.replace(source["short_url"], source["value"])
            unique_sources.append(source)

    return {
        "messages": [{"type": "ai", "content": content, "id": str(id(result))}],
        "sources_gathered": unique_sources,
    }


# ---------------------------------------------------------------------------
# Main async generator – yields SSE-ready event dicts
# ---------------------------------------------------------------------------

async def run_research_pipeline(
    messages: list,
    initial_search_query_count: int,
    max_research_loops: int,
    reasoning_model: str,
    config: Configuration,
) -> AsyncGenerator[dict, None]:
    """
    Run the full research pipeline and yield progress events.

    Yields dicts that map directly to frontend event types:
      generate_query, web_research, reflection, finalize_answer, messages
    """

    # 1. Generate initial search queries
    logger.info("Starting query generation")
    queries: list = await _rate_limited(
        _generate_queries_sync, messages, initial_search_query_count, config, reasoning_model
    )
    yield {"generate_query": {"search_query": queries}}

    # 2. Web research (serialized through rate limiter)
    all_sources: list = []
    all_research: list = []

    logger.info(f"Starting web research for {len(queries)} queries")
    for idx, query in enumerate(queries):
        result = await _rate_limited(_web_research_sync, query, idx, config, reasoning_model)
        all_sources.extend(result["sources_gathered"])
        all_research.append(result["web_research_result"])
        yield {"web_research": {"sources_gathered": result["sources_gathered"]}}

    # 3. Reflection + optional follow-up loops
    research_loop_count = 0
    query_count = len(queries)

    while True:
        logger.info(f"Starting reflection loop {research_loop_count + 1}")
        reflection = await _rate_limited(
            _reflect_sync, messages, all_research, reasoning_model, config
        )
        research_loop_count += 1
        yield {"reflection": reflection}

        if reflection["is_sufficient"] or research_loop_count >= max_research_loops:
            break

        # Follow-up research
        follow_up_queries = reflection.get("follow_up_queries", [])
        if not follow_up_queries:
            break

        logger.info(f"Starting follow-up research for {len(follow_up_queries)} queries")
        for idx, q in enumerate(follow_up_queries):
            result = await _rate_limited(_web_research_sync, q, query_count + idx, config, reasoning_model)
            all_sources.extend(result["sources_gathered"])
            all_research.append(result["web_research_result"])
            yield {"web_research": {"sources_gathered": result["sources_gathered"]}}
        query_count += len(follow_up_queries)

    # 4. Finalize answer
    logger.info("Finalizing answer")
    yield {"finalize_answer": True}

    answer = await _rate_limited(
        _finalize_answer_sync,
        messages,
        all_research,
        all_sources,
        reasoning_model,
        config,
    )
    yield {"messages": answer["messages"], "sources_gathered": answer["sources_gathered"]}
