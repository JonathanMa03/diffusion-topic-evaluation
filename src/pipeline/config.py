from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class PipelineConfig:
    project_root: Path
    start_year: int
    end_year: int

    db_path: Path
    data_path: Path
    outputs_path: Path
    logs_path: Path
    runs_path: Path
    run_dir: Path

    embedding_model_name: str = "all-MiniLM-L6-v2"
    topic_model_name: str = "kmeans_all-MiniLM-L6-v2_k15"

    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 2

    pubmed_query: str = ""
    run_name: str = ""


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "broad_biomedical"


def build_config(
    start_year: int,
    end_year: int,
    pubmed_query: str = "",
) -> PipelineConfig:
    project_root = Path.cwd().resolve()

    runs_path = project_root / "runs"
    runs_path.mkdir(parents=True, exist_ok=True)

    query_slug = _slugify(pubmed_query) if pubmed_query.strip() else "broad_biomedical"
    run_name = f"{start_year}_{end_year}_{query_slug}"
    run_dir = runs_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return PipelineConfig(
        project_root=project_root,
        start_year=start_year,
        end_year=end_year,
        db_path=project_root / "db" / "app.db",
        data_path=project_root / "data",
        outputs_path=project_root / "outputs",
        logs_path=project_root / "logs",
        runs_path=runs_path,
        run_dir=run_dir,
        pubmed_query=pubmed_query,
        run_name=run_name,
    )