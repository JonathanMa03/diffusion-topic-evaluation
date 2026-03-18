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

    import pickle

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split

    _check_file_exists(config.data_path, "Data directory")

    traj_path = config.data_path / "topic_trajectories.pkl"
    _check_file_exists(traj_path, "Topic trajectories artifact")

    future_csv_path = config.data_path / "future_topic_movement.csv"
    future_pkl_path = config.data_path / "future_topic_states.pkl"
    model_path = config.data_path / "diffusion_mlp.pt"

    print(f"  Using trajectories: {traj_path}")

    with open(traj_path, "rb") as f:
        topic_trajectories = pickle.load(f)

    if len(topic_trajectories) == 0:
        raise ValueError("Loaded topic trajectories are empty.")

    print(f"  Loaded {len(topic_trajectories)} topic trajectories")

    transition_rows = []

    for topic_id, info in topic_trajectories.items():
        years = info["years"]
        traj = info["trajectory"]
        label = info["label"]

        if len(years) != len(traj):
            raise ValueError(
                f"Topic {topic_id} has mismatched years ({len(years)}) "
                f"and trajectory length ({len(traj)})."
            )

        for t in range(len(years) - 1):
            transition_rows.append({
                "topic_id": topic_id,
                "topic_label": label,
                "year_from": years[t],
                "year_to": years[t + 1],
                "x_t": np.asarray(traj[t], dtype=np.float32),
                "x_t1": np.asarray(traj[t + 1], dtype=np.float32),
            })

    transitions_df = pd.DataFrame(transition_rows)

    if transitions_df.empty:
        raise ValueError("No transition pairs could be constructed from topic trajectories.")

    X_t = np.vstack(transitions_df["x_t"].values).astype(np.float32)
    X_t1 = np.vstack(transitions_df["x_t1"].values).astype(np.float32)

    print(f"  Built {len(transitions_df)} transition pairs")
    print(f"  Transition matrix shape: {X_t.shape}")

    noise_levels = np.array([0.01, 0.03, 0.05, 0.08, 0.12], dtype=np.float32)
    rng = np.random.default_rng(42)

    expanded_rows = []

    for i in range(len(X_t)):
        x_t = X_t[i]
        x_t1 = X_t1[i]
        topic_id = transitions_df.iloc[i]["topic_id"]
        topic_label = transitions_df.iloc[i]["topic_label"]
        year_from = transitions_df.iloc[i]["year_from"]
        year_to = transitions_df.iloc[i]["year_to"]

        for sigma in noise_levels:
            noise = rng.normal(
                loc=0.0,
                scale=float(sigma),
                size=x_t1.shape
            ).astype(np.float32)

            x_t1_noisy = x_t1 + noise

            expanded_rows.append({
                "topic_id": topic_id,
                "topic_label": topic_label,
                "year_from": year_from,
                "year_to": year_to,
                "sigma": float(sigma),
                "x_t": x_t,
                "x_t1_clean": x_t1,
                "x_t1_noisy": x_t1_noisy,
            })

    expanded_df = pd.DataFrame(expanded_rows)

    X_curr = np.vstack(expanded_df["x_t"].values).astype(np.float32)
    X_next_clean = np.vstack(expanded_df["x_t1_clean"].values).astype(np.float32)
    X_next_noisy = np.vstack(expanded_df["x_t1_noisy"].values).astype(np.float32)
    sigmas = expanded_df["sigma"].to_numpy(dtype=np.float32).reshape(-1, 1)

    print(f"  Expanded training set shape: {X_curr.shape}")

    idx = np.arange(len(X_curr))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    class TopicDiffusionDataset(Dataset):
        def __init__(self, X_curr, X_next_noisy, sigmas, X_next_clean):
            self.X_curr = torch.tensor(X_curr, dtype=torch.float32)
            self.X_next_noisy = torch.tensor(X_next_noisy, dtype=torch.float32)
            self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
            self.X_next_clean = torch.tensor(X_next_clean, dtype=torch.float32)

        def __len__(self):
            return len(self.X_curr)

        def __getitem__(self, idx):
            return {
                "x_curr": self.X_curr[idx],
                "x_next_noisy": self.X_next_noisy[idx],
                "sigma": self.sigmas[idx],
                "x_next_clean": self.X_next_clean[idx],
            }

    train_dataset = TopicDiffusionDataset(
        X_curr[train_idx],
        X_next_noisy[train_idx],
        sigmas[train_idx],
        X_next_clean[train_idx],
    )
    val_dataset = TopicDiffusionDataset(
        X_curr[val_idx],
        X_next_noisy[val_idx],
        sigmas[val_idx],
        X_next_clean[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    class DenoiserMLP(nn.Module):
        def __init__(self, dim=384, hidden_dim=512):
            super().__init__()
            input_dim = dim + dim + 1
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, x_curr, x_next_noisy, sigma):
            x = torch.cat([x_curr, x_next_noisy, sigma], dim=1)
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoiserMLP(dim=X_curr.shape[1], hidden_dim=512).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(loader, optimizer=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        total_loss = 0.0
        total_n = 0

        for batch in loader:
            x_curr = batch["x_curr"].to(device)
            x_next_noisy = batch["x_next_noisy"].to(device)
            sigma = batch["sigma"].to(device)
            x_next_clean = batch["x_next_clean"].to(device)

            with torch.set_grad_enabled(is_train):
                pred = model(x_curr, x_next_noisy, sigma)
                loss = criterion(pred, x_next_clean)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            batch_size = x_curr.size(0)
            total_loss += loss.item() * batch_size
            total_n += batch_size

        return total_loss / total_n

    print("  Training denoiser...")
    epochs = 100
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, optimizer=optimizer)
        val_loss = run_epoch(val_loader, optimizer=None)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

    model.eval()

    val_preds = []
    val_targets = []
    val_noisy = []

    with torch.no_grad():
        for batch in val_loader:
            x_curr = batch["x_curr"].to(device)
            x_next_noisy = batch["x_next_noisy"].to(device)
            sigma = batch["sigma"].to(device)
            x_next_clean = batch["x_next_clean"].to(device)

            pred = model(x_curr, x_next_noisy, sigma)

            val_preds.append(pred.cpu().numpy())
            val_targets.append(x_next_clean.cpu().numpy())
            val_noisy.append(x_next_noisy.cpu().numpy())

    Y_val_pred = np.vstack(val_preds)
    Y_val_true = np.vstack(val_targets)
    Y_val_noisy = np.vstack(val_noisy)

    mse_noisy = np.mean((Y_val_noisy - Y_val_true) ** 2)
    mse_pred = np.mean((Y_val_pred - Y_val_true) ** 2)

    print(f"  Validation noisy MSE:   {mse_noisy:.8f}")
    print(f"  Validation denoised MSE:{mse_pred:.8f}")

    latest_states = []

    for topic_id, info in topic_trajectories.items():
        years = info["years"]
        traj = info["trajectory"]

        if len(years) == 0:
            continue

        latest_states.append({
            "topic_id": topic_id,
            "topic_label": info["label"],
            "latest_year": years[-1],
            "x_latest": np.asarray(traj[-1], dtype=np.float32),
        })

    latest_df = pd.DataFrame(latest_states)
    future_df = latest_df[latest_df["latest_year"] == config.end_year].copy()

    if future_df.empty:
        raise ValueError(
            f"No topics found with latest_year == {config.end_year}. "
            "Cannot generate future predictions."
        )

    X_latest = np.vstack(future_df["x_latest"].values).astype(np.float32)

    future_sigma = 0.08
    rng_future = np.random.default_rng(42)

    X_future_noisy = X_latest + rng_future.normal(
        loc=0.0,
        scale=future_sigma,
        size=X_latest.shape
    ).astype(np.float32)

    sigma_future = np.full((len(X_latest), 1), future_sigma, dtype=np.float32)

    with torch.no_grad():
        x_curr_t = torch.tensor(X_latest, dtype=torch.float32, device=device)
        x_next_noisy_t = torch.tensor(X_future_noisy, dtype=torch.float32, device=device)
        sigma_t = torch.tensor(sigma_future, dtype=torch.float32, device=device)

        X_future_pred = model(x_curr_t, x_next_noisy_t, sigma_t).cpu().numpy()

    movement_norm = np.linalg.norm(X_future_pred - X_latest, axis=1)

    def cosine_sim(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return np.nan
        return float(np.dot(a, b) / denom)

    future_results_df = future_df[["topic_id", "topic_label", "latest_year"]].copy()
    future_results_df["future_sigma"] = future_sigma
    future_results_df["movement_norm"] = movement_norm
    future_results_df["cosine_similarity_to_latest"] = [
        cosine_sim(X_future_pred[i], X_latest[i])
        for i in range(len(X_latest))
    ]

    history_df = pd.DataFrame(history)

    print(f"  Saving model to: {model_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": X_curr.shape[1],
            "epochs": epochs,
            "history": history,
            "validation_noisy_mse": float(mse_noisy),
            "validation_denoised_mse": float(mse_pred),
        },
        model_path,
    )

    print(f"  Saving future movement CSV to: {future_csv_path}")
    future_results_df.to_csv(future_csv_path, index=False)

    future_state_artifact = {
        "topic_ids": future_df["topic_id"].tolist(),
        "topic_labels": future_df["topic_label"].tolist(),
        "latest_year": future_df["latest_year"].tolist(),
        "x_latest": X_latest,
        "x_future_noisy": X_future_noisy,
        "x_future_pred": X_future_pred,
        "future_sigma": future_sigma,
        "embedding_dim": X_latest.shape[1],
        "history": history_df,
        "validation_noisy_mse": float(mse_noisy),
        "validation_denoised_mse": float(mse_pred),
    }

    print(f"  Saving future topic states to: {future_pkl_path}")
    with open(future_pkl_path, "wb") as f:
        pickle.dump(future_state_artifact, f)

    print("  Diffusion artifacts saved successfully")


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

    print("  Saving movement plot...")
    plot_df = future_results_df.sort_values("movement_norm", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["topic_label"], plot_df["movement_norm"])
    plt.xlabel("Predicted movement norm")
    plt.ylabel("Topic")
    plt.title("Predicted 2023 Topic Movement (Neural Denoiser)")
    plt.tight_layout()
    plt.savefig(config.outputs_path / "movement_norm.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saving cosine similarity plot...")
    plot_df = future_results_df.sort_values("cosine_similarity_to_latest", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["topic_label"], plot_df["cosine_similarity_to_latest"])
    plt.xlabel("Cosine similarity to 2022 state")
    plt.ylabel("Topic")
    plt.title("Predicted 2023 Stability Relative to 2022")
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