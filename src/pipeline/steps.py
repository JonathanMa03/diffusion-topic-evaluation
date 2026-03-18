# src/pipeline/steps.py

from pathlib import Path


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")


def step_ingestion(config):
    print("[STEP] Ingestion")
    print(f"  Target years: {config.start_year} → {config.end_year}")
    print("  Status: not implemented yet")

    raise NotImplementedError(
        "Ingestion step is not implemented yet. "
        "Next action: move notebook 01 data ingestion logic into step_ingestion(config)."
    )


def step_embeddings(config):
    print("[STEP] Embeddings")
    _check_file_exists(config.db_path, "Database")

    print(f"  Using DB: {config.db_path}")
    print(f"  Embedding model: {config.embedding_model_name}")
    print("  Status: not implemented yet")

    raise NotImplementedError(
        "Embedding step is not implemented yet. "
        "Next action: move notebook 02 embedding pipeline logic into step_embeddings(config)."
    )


def step_topics(config):
    print("[STEP] Topic Discovery")
    _check_file_exists(config.db_path, "Database")

    print(f"  Using DB: {config.db_path}")
    print(f"  HDBSCAN min_cluster_size: {config.hdbscan_min_cluster_size}")
    print(f"  HDBSCAN min_samples: {config.hdbscan_min_samples}")
    print("  Status: not implemented yet")

    raise NotImplementedError(
        "Topic step is not implemented yet. "
        "Next action: move notebook 03/04/07 topic discovery and lineage logic into step_topics(config)."
    )


def step_diffusion(config):
    print("[STEP] Diffusion Modeling")
    _check_file_exists(config.data_path, "Data directory")

    traj_path = config.data_path / "topic_trajectories.pkl"
    _check_file_exists(traj_path, "Topic trajectories artifact")

    print(f"  Using trajectories: {traj_path}")
    print("  Status: not implemented yet")

    raise NotImplementedError(
        "Diffusion step is not implemented yet. "
        "Next action: move notebook 05/06 diffusion logic into step_diffusion(config)."
    )


def step_visualizations(config):
    print("[STEP] Visualizations")

    import pickle
    import textwrap

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _check_file_exists(config.data_path, "Data directory")

    traj_path = config.data_path / "topic_trajectories.pkl"
    future_path = config.data_path / "future_topic_movement.csv"
    lineage_path = config.data_path / "hdbscan_lineage.csv"

    required_inputs = [traj_path, future_path, lineage_path]
    for path in required_inputs:
        _check_file_exists(path, "Visualization input artifact")

    config.outputs_path.mkdir(parents=True, exist_ok=True)

    print("  Loading artifacts...")
    with open(traj_path, "rb") as f:
        topic_trajectories = pickle.load(f)

    future_results_df = pd.read_csv(future_path)
    lineage_df = pd.read_csv(lineage_path)

    print(f"   - trajectories: {len(topic_trajectories)} topics")
    print(f"   - future movement rows: {len(future_results_df)}")
    print(f"   - lineage rows: {len(lineage_df)}")

    def wrap_label(s, width=24):
        return "\n".join(textwrap.wrap(s, width=width))

    year_markers = {
        2020: "o",
        2021: "s",
        2022: "^",
    }

    movement_lookup = future_results_df.set_index("topic_id")["movement_norm"].to_dict()

    print("  Building PCA trajectory data...")
    all_points = []
    meta = []

    for topic_id, info in topic_trajectories.items():
        traj = info["trajectory"]
        years = info["years"]
        label = info["label"]

        for i, year in enumerate(years):
            all_points.append(traj[i])
            meta.append({
                "topic_id": topic_id,
                "label": label,
                "year": year,
                "movement_norm": movement_lookup.get(topic_id, np.nan),
            })

    if len(all_points) == 0:
        raise ValueError("No trajectory points found for PCA visualization.")

    X_all = np.vstack(all_points)
    meta_df = pd.DataFrame(meta)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_all)

    meta_df["x"] = X_2d[:, 0]
    meta_df["y"] = X_2d[:, 1]

    print("  Saving movement histogram...")
    plt.figure(figsize=(9, 6))
    plt.hist(future_results_df["movement_norm"], bins=10)
    plt.xlabel("Movement norm")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Topic Movement")
    plt.tight_layout()
    plt.savefig(config.outputs_path / "movement_norm.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saving cosine similarity histogram...")
    plt.figure(figsize=(9, 6))
    plt.hist(future_results_df["cosine_similarity_to_latest"], bins=10)
    plt.xlabel("Cosine similarity to latest state")
    plt.ylabel("Count")
    plt.title("Cosine Similarity to Latest Topic State")
    plt.tight_layout()
    plt.savefig(config.outputs_path / "cosine_sim_latest.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saving top-8 PCA trajectories...")
    TOP_K = 8
    top_topic_ids = (
        future_results_df
        .sort_values("movement_norm", ascending=False)
        .head(TOP_K)["topic_id"]
        .tolist()
    )

    top_meta_df = meta_df[meta_df["topic_id"].isin(top_topic_ids)].copy()

    plt.figure(figsize=(13, 9))
    cmap = plt.cm.get_cmap("tab10", len(top_topic_ids))

    for i, topic_id in enumerate(top_topic_ids):
        group = top_meta_df[top_meta_df["topic_id"] == topic_id].sort_values("year")

        x = group["x"].values
        y = group["y"].values
        label = group["label"].iloc[0]
        movement = group["movement_norm"].iloc[0]
        color = cmap(i)

        lw = 2 + 2.5 * movement

        plt.plot(x, y, linewidth=lw, alpha=0.9, color=color)

        for _, row in group.iterrows():
            plt.scatter(
                row["x"], row["y"],
                marker=year_markers.get(row["year"], "o"),
                s=100,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=3
            )

        for j in range(len(group) - 1):
            plt.annotate(
                "",
                xy=(x[j + 1], y[j + 1]),
                xytext=(x[j], y[j]),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=lw * 0.6,
                    color=color,
                    alpha=0.8
                )
            )

        last = group.iloc[-1]
        dx = 10 if last["x"] < 0.3 else -10
        dy = 8 if last["y"] < 0 else -8

        plt.annotate(
            wrap_label(label, width=22),
            xy=(last["x"], last["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec=color,
                alpha=0.95
            )
        )

    for year, marker in year_markers.items():
        plt.scatter([], [], marker=marker, s=90, label=str(year), color="black")

    plt.title("Top 8 Moving Topic Trajectories (PCA Projection)", fontsize=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Year", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(config.outputs_path / "top8_pca.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saving lineage plot...")
    lineage_summary = (
        lineage_df.groupby("lineage_id")
        .agg(
            start_year=("year", "min"),
            end_year=("year", "max"),
            n_years=("year", "nunique"),
            total_docs=("n_docs", "sum"),
        )
        .reset_index()
    )

    def classify_lineage(row):
        if row["n_years"] >= 2:
            return "persistent"
        elif row["start_year"] == 2020 and row["end_year"] == 2020:
            return "dies_after_2020"
        elif row["start_year"] == 2021 and row["end_year"] == 2021:
            return "born_in_2021"
        else:
            return "other"

    lineage_summary["status"] = lineage_summary.apply(classify_lineage, axis=1)

    lineage_name_map = {
        0: "Lineage 0: Medical imaging cluster",
        1: "Lineage 1: Mobility and network spread cluster",
        2: "Lineage 2: Cross-sectional public health cluster",
        3: "Lineage 3: Main clinical care cluster",
        4: "Lineage 4: Digital media and infodemic cluster",
        5: "Lineage 5: Online biomedical education cluster",
        6: "Lineage 6: Financial markets cluster",
    }

    lineage_name_df = pd.DataFrame(
        [{"lineage_id": k, "lineage_name": v} for k, v in lineage_name_map.items()]
    )

    lineage_plot_df = lineage_df.merge(
        lineage_summary[["lineage_id", "status", "total_docs"]],
        on="lineage_id",
        how="left"
    ).merge(
        lineage_name_df,
        on="lineage_id",
        how="left"
    )

    lineage_plot_df["lineage_name"] = lineage_plot_df["lineage_name"].fillna(
        lineage_plot_df["lineage_id"].apply(lambda x: f"Lineage {int(x)}")
    )

    status_colors = {
        "persistent": "tab:blue",
        "dies_after_2020": "tab:red",
        "born_in_2021": "tab:green",
        "other": "gray",
    }

    plt.figure(figsize=(14, 7))

    for lineage_id, group in lineage_plot_df.groupby("lineage_id"):
        group = group.sort_values("year")

        status = group["status"].iloc[0]
        color = status_colors.get(status, "gray")

        lineage_name = group["lineage_name"].iloc[0]
        total_docs = int(group["total_docs"].iloc[0])

        x = group["year"].values
        y = np.full(len(group), lineage_id)

        line_width = 3 if status == "persistent" else 2

        if len(group) > 1:
            plt.plot(x, y, color=color, linewidth=line_width, alpha=0.9)

        plt.scatter(
            x,
            y,
            s=group["n_docs"].values * 7,
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )

        for _, row in group.iterrows():
            plt.annotate(
                f'n={int(row["n_docs"])}',
                xy=(row["year"], lineage_id),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=color
            )

        plt.annotate(
            f"{lineage_name} (total n={total_docs})",
            xy=(2019.74, lineage_id),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=color,
                alpha=0.92
            )
        )

    plt.xlim(2019.55, 2021.08)
    plt.xticks([2020, 2021])
    plt.xlabel("Year")
    plt.ylabel("Lineage ID")
    plt.title("Dynamic Topic Lineages (HDBSCAN + Semantic Labels)", fontsize=14)
    plt.grid(axis="x", alpha=0.2)
    plt.subplots_adjust(left=0.38)
    plt.savefig(config.outputs_path / "linneage.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved outputs:")
    print(f"   - {config.outputs_path / 'movement_norm.png'}")
    print(f"   - {config.outputs_path / 'cosine_sim_latest.png'}")
    print(f"   - {config.outputs_path / 'top8_pca.png'}")
    print(f"   - {config.outputs_path / 'linneage.png'}")