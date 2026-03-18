from src.pipeline.config import build_config
from src.pipeline.steps import step_embeddings


def main():
    config = build_config(start_year=2020, end_year=2022)

    try:
        step_embeddings(config)
        print("\n Embedding step completed successfully")
    except Exception as e:
        print("\n Embedding step failed")
        print(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()