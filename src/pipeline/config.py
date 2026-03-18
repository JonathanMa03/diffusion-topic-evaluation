from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    project_root: Path
    start_year: int
    end_year: int

    db_path: Path
    data_path: Path
    outputs_path: Path
    logs_path: Path

    embedding_model_name: str = "all-MiniLM-L6-v2"
    topic_model_name: str = "kmeans_all-MiniLM-L6-v2_k15"

    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 2


def build_config(start_year: int, end_year: int) -> PipelineConfig:
    project_root = Path.cwd().resolve()

    return PipelineConfig(
        project_root=project_root,
        start_year=start_year,
        end_year=end_year,
        db_path=project_root / "db" / "app.db",
        data_path=project_root / "data",
        outputs_path=project_root / "outputs",
        logs_path=project_root / "logs",
    )