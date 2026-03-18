from src.pipeline.config import build_config
from src.pipeline.steps import step_ingestion


def main():
    # blank pubmed_query = broad biomedical literature in the selected range
    config = build_config(
        start_year=2020,
        end_year=2022,
        pubmed_query="",
    )

    try:
        step_ingestion(config)
        print("\n Ingestion step completed successfully")
    except Exception as e:
        print("\n Ingestion step failed")
        print(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()