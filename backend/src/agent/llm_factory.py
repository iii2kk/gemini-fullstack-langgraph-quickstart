import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.configuration import Configuration

DEFAULT_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]


class LLMFactory:
    """Create chat model clients from configuration."""

    def __init__(self, config: Configuration) -> None:
        self._config = config

    def create_chat_model(
        self,
        model: str,
        *,
        temperature: float,
        max_retries: int = 0,
    ) -> Any:
        """Create a chat model client for the configured provider."""
        provider = self._config.llm_provider

        if provider == "gemini":
            api_key = self._config.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set")

            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_retries=max_retries,
                api_key=api_key,
            )

        if provider == "openai_compatible":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ImportError(
                    "langchain_openai is required when llm_provider='openai_compatible'"
                ) from exc

            base_url = self._config.openai_compatible_base_url or os.getenv(
                "OPENAI_COMPATIBLE_BASE_URL"
            )
            if not base_url:
                raise ValueError("OPENAI_COMPATIBLE_BASE_URL is not set")

            api_key = (
                self._config.openai_compatible_api_key
                or os.getenv("OPENAI_COMPATIBLE_API_KEY")
                or "dummy"
            )

            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_retries=max_retries,
                api_key=api_key,
                base_url=base_url,
            )

        raise ValueError(f"Unsupported llm_provider: {provider}")

    def list_models(self) -> list[str]:
        """List available model ids for the configured provider."""
        provider = self._config.llm_provider

        if provider == "gemini":
            return DEFAULT_GEMINI_MODELS.copy()

        if provider == "openai_compatible":
            llm = self.create_chat_model(
                model=self._config.answer_model,
                temperature=0,
                max_retries=0,
            )
            response = llm.root_client.models.list()
            data = getattr(response, "data", response)
            return sorted(model.id for model in data)

        raise ValueError(f"Unsupported llm_provider: {provider}")
