import os
from pydantic import BaseModel, Field
from typing import Any, Optional


class Configuration(BaseModel):
    """The configuration for the agent."""

    search_provider: str = Field(
        default="gemini_google_search",
        metadata={
            "description": "The web search provider to use. Supported values: gemini_google_search, searxng, 4get."
        },
    )

    searxng_base_url: str | None = Field(
        default=None,
        metadata={"description": "Base URL for a SearxNG instance, for example http://localhost:8080."},
    )

    searxng_result_limit: int = Field(
        default=8,
        metadata={"description": "Maximum number of SearxNG search results to use for synthesis."},
    )

    searxng_language: str = Field(
        default="all",
        metadata={"description": "Language parameter for SearxNG search, for example ja, en, or all."},
    )

    fourget_base_url: str = Field(
        default="https://4get.ca",
        metadata={"description": "Base URL for a 4get instance, for example https://4get.ca."},
    )

    fourget_result_limit: int = Field(
        default=8,
        metadata={"description": "Maximum number of 4get web results to use for synthesis."},
    )

    fourget_country: str | None = Field(
        default=None,
        metadata={"description": "Optional 4get country filter for web search, for example ja_JP."},
    )

    fourget_extended_search: bool = Field(
        default=False,
        metadata={"description": "Whether to enable 4get extendedsearch for richer result data."},
    )

    fourget_pass_token: str | None = Field(
        default=None,
        metadata={"description": "Optional 4get pass token sent as a cookie named 'pass'."},
    )

    llm_provider: str = Field(
        default="gemini",
        metadata={
            "description": "The LLM provider to use. Supported values: gemini, openai_compatible."
        },
    )

    gemini_api_key: str | None = Field(
        default=None,
        metadata={"description": "Optional Gemini API key override."},
    )

    openai_compatible_base_url: str | None = Field(
        default=None,
        metadata={"description": "Base URL for an OpenAI-compatible local LLM server."},
    )

    openai_compatible_api_key: str | None = Field(
        default=None,
        metadata={"description": "Optional API key for an OpenAI-compatible server."},
    )

    query_generator_model: str = Field(
        default="gemini-2.5-flash",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-pro",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_config(
        cls, config: Optional[dict] = None
    ) -> "Configuration":
        """Create a Configuration instance from a config dict."""
        config = config or {}

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), config.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
