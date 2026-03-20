from abc import ABC, abstractmethod

from agent.configuration import Configuration


class BaseSearchProvider(ABC):
    """Abstract base class for search providers."""

    def __init__(self, config: Configuration) -> None:
        self._config = config

    @abstractmethod
    def web_research(
        self,
        search_query: str,
        idx: int,
        *,
        model: str,
        logger,
        sep: str,
    ) -> dict:
        """Run web research for a query and return the normalized result."""
