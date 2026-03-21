from agent.search_providers.base import BaseSearchProvider
from agent.search_providers.fourget_search import FourGetSearchProvider
from agent.search_providers.gemini_google_search import GeminiGoogleSearchProvider
from agent.search_providers.searxng_search import SearxngSearchProvider


def create_search_provider(config) -> BaseSearchProvider:
    """Create a search provider from configuration."""
    if config.search_provider == "gemini_google_search":
        return GeminiGoogleSearchProvider(config)
    if config.search_provider == "searxng":
        return SearxngSearchProvider(config)
    if config.search_provider == "4get":
        return FourGetSearchProvider(config)

    raise ValueError(f"Unsupported search_provider: {config.search_provider}")
