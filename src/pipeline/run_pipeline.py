# src/pipeline/run_pipeline.py

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

    print("\n===== PIPELINE START =====")
    print(f"Years: {config.start_year} → {config.end_year}")
    print(f"DB: {config.db_path}")
    print(f"Outputs: {config.outputs_path}")

    for i, (step_name, step_fn) in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] {step_name}")

        try:
            step_fn(config)

        except Exception as e:
            print("\n PIPELINE FAILED")
            print(f"Stage: {step_name}")
            print(f"Error: {str(e)}")

            # ensure logs directory exists
            config.logs_path.mkdir(exist_ok=True)

            # create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = config.logs_path / f"pipeline_error_{timestamp}.log"

            with open(log_file, "w") as f:
                f.write("PIPELINE FAILURE\n")
                f.write(f"Stage: {step_name}\n\n")
                f.write(traceback.format_exc())

            print(f"Full traceback saved to: {log_file}")

            return False

    print("\n PIPELINE COMPLETED SUCCESSFULLY")
    return True