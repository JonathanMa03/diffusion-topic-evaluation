# Diffusion-Based Topic Evolution in Biomedical Literature

This project models how biomedical research topics evolve over time by combining topic discovery with diffusion modeling in latent embedding space.

Rather than fixing a static set of topics, the workflow extracts topic representations from biomedical abstracts grouped by time period, aligns related topics across years, and then learns their semantic drift using a diffusion model on topic embeddings. This makes it possible to study topic persistence, emergence, decline, and forecast plausible future topic states.

The initial development workflow is notebook-first:
- data ingestion and preprocessing in Jupyter notebooks
- topic extraction and alignment experiments in notebooks
- diffusion modeling experiments in notebooks
- later migration of stable components into Python modules
- dashboard/application layer built afterward with `.py` files

A SQLite database is used to store document metadata, cleaned text, embeddings, topic assignments, topic trajectories, and experiment outputs.

## Project goals

- build a biomedical literature pipeline using PubMed, BioASQ, CORD-19, or a curated subset
- extract topic representations by time period
- align semantically similar topics across adjacent years
- train a diffusion model on topic embeddings
- visualize topic trajectories and semantic drift
- optionally forecast future topic movement in embedding space

## Core idea

Pipeline:

1. collect biomedical abstracts with publication dates
2. preprocess and store documents in SQLite
3. compute document embeddings using biomedical or scientific language models
4. discover topics for each year or time block
5. represent each topic by an embedding centroid or learned topic embedding
6. align topics across time into trajectories
7. train a diffusion model on topic embedding trajectories
8. visualize evolution and later deploy a dashboard

## Planned stack

- Python
- Jupyter notebooks
- SQLite
- pandas / numpy
- scikit-learn
- sentence-transformers
- PyTorch
- matplotlib / plotly
- SQLAlchemy

Optional later additions:
- SciBERT / BioBERT-based embeddings
- UMAP / HDBSCAN
- Dash or Streamlit dashboard
- experiment tracking with MLflow

## Repository structure

```text
diffusion-topic-evolution/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── db/
│   ├── app.db
│   ├── schema.sql
│   └── migrations/
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_preprocessing_and_eda.ipynb
│   ├── 03_embedding_pipeline.ipynb
│   ├── 04_topic_discovery_baselines.ipynb
│   ├── 05_topic_alignment.ipynb
│   ├── 06_diffusion_topic_evolution.ipynb
│   ├── 07_evaluation_and_visualization.ipynb
│   └── 99_scratch.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── models.py
│   │   ├── schema.py
│   │   └── crud.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── clean.py
│   │   ├── tokenize.py
│   │   └── loaders.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── encode.py
│   │   ├── pooling.py
│   │   └── reduction.py
│   ├── topics/
│   │   ├── __init__.py
│   │   ├── discover.py
│   │   ├── represent.py
│   │   ├── align.py
│   │   └── trajectories.py
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── sample.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── coherence.py
│   │   ├── alignment_metrics.py
│   │   └── drift_metrics.py
│   └── visualization/
│       ├── __init__.py
│       ├── topics.py
│       ├── timelines.py
│       └── embeddings.py
├── scripts/
│   ├── init_db.py
│   ├── ingest_pubmed.py
│   ├── build_embeddings.py
│   ├── run_topic_pipeline.py
│   └── train_diffusion.py
├── dashboards/
│   ├── app.py
│   ├── pages/
│   └── components/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── reports/
├── tests/
│   ├── test_database.py
│   ├── test_embeddings.py
│   ├── test_alignment.py
│   └── test_diffusion.py
└── docs/
    ├── notes/
    ├── references/
    └── roadmap.md
```
