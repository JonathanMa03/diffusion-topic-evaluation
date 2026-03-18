from src.pipeline.config import build_config
from src.pipeline.run_pipeline import run_full_pipeline


def main():
    # blank pubmed_query = broad biomedical literature in the selected range
    config = build_config(
        start_year=2020,
        end_year=2022,
        pubmed_query="",
    )

    success = run_full_pipeline(config)

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()