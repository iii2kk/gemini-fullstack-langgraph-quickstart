import logging

from dotenv import load_dotenv

from agent.configuration import Configuration
from agent.search_providers import create_search_provider


def main() -> None:
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_search_provider")

    config = Configuration.from_config()
    provider = create_search_provider(config)

    query = "Physical AI future outlook 2026"
    result = provider.web_research(
        query,
        0,
        model=config.query_generator_model,
        logger=logger,
        sep="=" * 60,
    )

    print("provider:", type(provider).__name__)
    print("query:", result["search_query"])
    print("sources:", len(result["sources_gathered"]))
    print("summary:")
    print(result["web_research_result"])


if __name__ == "__main__":
    main()
