import json
import traceback
from datetime import datetime

from .steps import (
    step_ingestion,
    step_embeddings,
    step_topics,
    step_diffusion,
    step_visualizations,
)


def run_full_pipeline(config):
    """
    Runs the full pipeline in order with strong error handling.
    Returns True if successful, False otherwise.
    """

    steps = [
        ("Ingestion", step_ingestion),
        ("Embeddings", step_embeddings),
        ("Topic Discovery", step_topics),
        ("Diffusion Modeling", step_diffusion),
        ("Visualization", step_visualizations),
    ]

    config.logs_path.mkdir(parents=True, exist_ok=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "run_name": config.run_name,
        "start_year": config.start_year,
        "end_year": config.end_year,
        "pubmed_query": config.pubmed_query,
        "embedding_model_name": config.embedding_model_name,
        "topic_model_name": config.topic_model_name,
        "hdbscan_min_cluster_size": config.hdbscan_min_cluster_size,
        "hdbscan_min_samples": config.hdbscan_min_samples,
        "started_at": datetime.now().isoformat(),
        "status": "running",
        "failed_stage": None,
    }

    metadata_file = config.run_dir / "run_metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(run_metadata, f, indent=2)

    print("\n===== PIPELINE START =====")
    print(f"Run name: {config.run_name}")
    print(f"Years: {config.start_year} → {config.end_year}")
    print(f"Query: {config.pubmed_query if config.pubmed_query else '[broad biomedical]'}")
    print(f"DB: {config.db_path}")
    print(f"Outputs: {config.outputs_path}")
    print(f"Run dir: {config.run_dir}")

    for i, (step_name, step_fn) in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] {step_name}")

        try:
            step_fn(config)

        except Exception as e:
            print("\n PIPELINE FAILED")
            print(f"Stage: {step_name}")
            print(f"Error: {str(e)}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = config.logs_path / f"pipeline_error_{timestamp}.log"

            with open(log_file, "w") as f:
                f.write("PIPELINE FAILURE\n")
                f.write(f"Run name: {config.run_name}\n")
                f.write(f"Stage: {step_name}\n\n")
                f.write(traceback.format_exc())

            run_metadata["status"] = "failed"
            run_metadata["failed_stage"] = step_name
            run_metadata["error_log"] = str(log_file)
            run_metadata["finished_at"] = datetime.now().isoformat()

            with open(metadata_file, "w") as f:
                json.dump(run_metadata, f, indent=2)

            print(f"Full traceback saved to: {log_file}")
            return False

    run_metadata["status"] = "completed"
    run_metadata["finished_at"] = datetime.now().isoformat()

    with open(metadata_file, "w") as f:
        json.dump(run_metadata, f, indent=2)

    print("\n PIPELINE COMPLETED SUCCESSFULLY")
    return True