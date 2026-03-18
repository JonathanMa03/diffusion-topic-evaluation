from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = PROJECT_ROOT / "runs"

TOP_K = 8

app = Dash(__name__)
app.title = "How Has Biomedical Research Evolved Over Time?"


def get_available_runs():
    if not RUNS_PATH.exists():
        return []

    runs = [d.name for d in RUNS_PATH.iterdir() if d.is_dir()]
    return sorted(runs, reverse=True)


def load_future_df(run_name: str) -> pd.DataFrame:
    path = RUNS_PATH / run_name / "data" / "future_topic_movement.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_trajectories(run_name: str):
    path = RUNS_PATH / run_name / "data" / "topic_trajectories.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_lineage_df(run_name: str) -> pd.DataFrame:
    path = RUNS_PATH / run_name / "data" / "hdbscan_lineage.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def artifact_status(run_name: str) -> dict:
    run_dir = RUNS_PATH / run_name
    return {
        "future_movement": (run_dir / "data" / "future_topic_movement.csv").exists(),
        "trajectories": (run_dir / "data" / "topic_trajectories.pkl").exists(),
        "lineage": (run_dir / "data" / "hdbscan_lineage.csv").exists(),
        "metadata": (run_dir / "run_metadata.json").exists(),
    }


def status_badge(label: str, ok: bool) -> html.Div:
    color = "#15803d" if ok else "#b91c1c"
    text = "available" if ok else "missing"
    return html.Div(
        [
            html.Span(f"{label}: ", style={"fontWeight": "600"}),
            html.Span(text, style={"color": color}),
        ],
        style={"marginBottom": "6px"},
    )


def build_figures(run_name: str):
    future_df = load_future_df(run_name)
    topic_trajectories = load_trajectories(run_name)
    lineage_df = load_lineage_df(run_name)

    movement_fig = go.Figure()
    cosine_fig = go.Figure()
    pca_fig = go.Figure()
    lineage_fig = go.Figure()

    # Movement + stability
    if not future_df.empty:
        movement_fig = px.bar(
            future_df.sort_values("movement_norm", ascending=True),
            x="movement_norm",
            y="topic_label",
            orientation="h",
            title="Predicted Topic Movement (Next Time Step)",
            labels={
                "movement_norm": "Predicted movement norm",
                "topic_label": "Topic",
            },
        )
        movement_fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        cosine_fig = px.bar(
            future_df.sort_values("cosine_similarity_to_latest", ascending=True),
            x="cosine_similarity_to_latest",
            y="topic_label",
            orientation="h",
            title="Topic Stability Relative to Latest Observed Period",
            labels={
                "cosine_similarity_to_latest": "Cosine similarity to latest period",
                "topic_label": "Topic",
            },
        )
        cosine_fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=20, r=20, t=60, b=20),
        )

    # PCA
    if topic_trajectories is not None and not future_df.empty:
        movement_lookup = future_df.set_index("topic_id")["movement_norm"].to_dict()

        top_topic_ids = (
            future_df.sort_values("movement_norm", ascending=False)
            .head(TOP_K)["topic_id"]
            .tolist()
        )

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

        if len(all_points) > 0:
            X_all = np.vstack(all_points)
            meta_df = pd.DataFrame(meta)

            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_all)

            meta_df["x"] = X_2d[:, 0]
            meta_df["y"] = X_2d[:, 1]

            top_meta_df = meta_df[meta_df["topic_id"].isin(top_topic_ids)].copy()

            pca_fig = go.Figure()

            for topic_id in top_topic_ids:
                group = top_meta_df[top_meta_df["topic_id"] == topic_id].sort_values("year")
                if group.empty:
                    continue

                label = group["label"].iloc[0]
                movement_value = group["movement_norm"].iloc[0]
                movement = float(movement_value) if not pd.isna(movement_value) else 0.0

                x = group["x"].values
                y = group["y"].values

                pca_fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=label,
                    line=dict(width=2 + 2.5 * movement),
                    hoverinfo="skip"
                ))

                marker_sizes = np.linspace(8, 14, len(x)).tolist()

                pca_fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=marker_sizes),
                    customdata=group[["year"]].values,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Year: %{customdata[0]}<br>"
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<br>"
                        f"Movement norm: {movement:.3f}<extra></extra>"
                    ),
                ))

                for j in range(len(x) - 1):
                    pca_fig.add_annotation(
                        x=x[j + 1],
                        y=y[j + 1],
                        ax=x[j],
                        ay=y[j],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.1,
                        arrowwidth=1.4,
                        opacity=0.7
                    )

            pca_fig.update_layout(
                title="Top Topic Trajectories (PCA Projection)",
                xaxis_title="PC1",
                yaxis_title="PC2",
                template="plotly_white",
                height=800,
                margin=dict(l=20, r=20, t=60, b=20),
                legend_title="Topic",
            )

    # Lineage
    if not lineage_df.empty:
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

        min_year = lineage_df["year"].min()
        max_year = lineage_df["year"].max()

        def classify_lineage(row):
            if row["n_years"] >= 2:
                return "persistent"
            elif row["end_year"] == row["start_year"] == min_year:
                return "early_only"
            elif row["end_year"] == row["start_year"] == max_year:
                return "late_only"
            else:
                return "other"

        lineage_summary["status"] = lineage_summary.apply(classify_lineage, axis=1)

        lineage_name_df = pd.DataFrame({
            "lineage_id": sorted(lineage_df["lineage_id"].unique())
        })
        lineage_name_df["lineage_name"] = lineage_name_df["lineage_id"].apply(
            lambda x: f"Lineage {int(x)}"
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

        lineage_plot_df["hover_text"] = (
            "<b>" + lineage_plot_df["lineage_name"] + "</b><br>"
            + "Year: " + lineage_plot_df["year"].astype(str) + "<br>"
            + "Cluster ID: " + lineage_plot_df["cluster_id"].astype(str) + "<br>"
            + "Docs in node: " + lineage_plot_df["n_docs"].astype(str) + "<br>"
            + "Total docs in lineage: " + lineage_plot_df["total_docs"].astype(str) + "<br>"
            + "Status: " + lineage_plot_df["status"].str.replace("_", " ", regex=False)
        )

        status_colors = {
            "persistent": "blue",
            "early_only": "red",
            "late_only": "green",
            "other": "gray",
        }

        lineage_fig = go.Figure()

        for lineage_id, group in lineage_plot_df.groupby("lineage_id"):
            group = group.sort_values("year")

            status_name = group["status"].iloc[0]
            color = status_colors.get(status_name, "gray")
            lineage_name = group["lineage_name"].iloc[0]

            x = group["year"].values
            y = [lineage_id] * len(group)

            lineage_fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=lineage_name,
                line=dict(color=color, width=3 if status_name == "persistent" else 2),
                hoverinfo="skip"
            ))

            lineage_fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                showlegend=False,
                text=[f"n={n}" for n in group["n_docs"]],
                textposition="top center",
                hovertext=group["hover_text"],
                hoverinfo="text",
                marker=dict(
                    color=color,
                    size=group["n_docs"] * 0.45 + 10,
                    line=dict(color="black", width=0.5)
                )
            ))

            for i in range(len(x) - 1):
                lineage_fig.add_annotation(
                    x=x[i + 1],
                    y=y[i + 1],
                    ax=x[i],
                    ay=y[i],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.2,
                    arrowwidth=1.5,
                    opacity=0.7
                )

        lineage_fig.update_layout(
            title="Interactive Dynamic Topic Lineages",
            xaxis_title="Year",
            yaxis_title="Lineage ID",
            template="plotly_white",
            height=750,
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title="Lineage",
            xaxis=dict(title="Year", dtick=1),
        )

    return movement_fig, cosine_fig, pca_fig, lineage_fig


available_runs = get_available_runs()
default_run = available_runs[0] if available_runs else None

app.layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        html.H1("Diffusion Topic Evolution Dashboard"),
        html.P(
            "Interactive view of topic movement, topic stability, trajectory geometry, "
            "and lineage structure."
        ),

        html.Div([
            html.Label("Select Run:"),
            dcc.Dropdown(
                id="run-selector",
                options=[{"label": r, "value": r} for r in available_runs],
                value=default_run,
                clearable=False,
            )
        ], style={"marginBottom": "20px"}),

        dcc.Tabs(
            children=[
                dcc.Tab(
                    label="Overview",
                    children=[
                        html.Div(
                            id="overview-content",
                            style={"padding": "24px 8px"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="Movement & Stability",
                    children=[
                        html.Div(
                            style={"padding": "24px 8px"},
                            children=[
                                html.H2("Predicted Topic Movement"),
                                dcc.Graph(id="movement-graph"),
                                html.H2("Predicted Topic Stability"),
                                dcc.Graph(id="cosine-graph"),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="PCA Trajectories",
                    children=[
                        html.Div(
                            style={"padding": "24px 8px"},
                            children=[
                                html.H2(f"Top {TOP_K} Topic Trajectories"),
                                dcc.Graph(id="pca-graph"),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Topic Lineages",
                    children=[
                        html.Div(
                            style={"padding": "24px 8px"},
                            children=[
                                html.H2("Topic Lineages"),
                                dcc.Graph(id="lineage-graph"),
                            ],
                        )
                    ],
                ),
            ]
        ),
    ],
)


@app.callback(
    Output("overview-content", "children"),
    Output("movement-graph", "figure"),
    Output("cosine-graph", "figure"),
    Output("pca-graph", "figure"),
    Output("lineage-graph", "figure"),
    Input("run-selector", "value"),
)
def update_dashboard(run_name):
    if not run_name:
        empty = go.Figure()
        return html.P("No runs available."), empty, empty, empty, empty

    status = artifact_status(run_name)
    run_dir = RUNS_PATH / run_name

    movement_fig, cosine_fig, pca_fig, lineage_fig = build_figures(run_name)

    overview = [
        html.H2("Current Status"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(f"Run: {run_name}"),
                        html.Div(f"Run path: {run_dir}"),
                    ],
                    style={"marginBottom": "12px"},
                ),
                status_badge("Future movement CSV", status["future_movement"]),
                status_badge("Topic trajectories PKL", status["trajectories"]),
                status_badge("HDBSCAN lineage CSV", status["lineage"]),
                status_badge("Run metadata JSON", status["metadata"]),
            ],
            style={
                "padding": "16px",
                "border": "1px solid #ddd",
                "borderRadius": "10px",
                "backgroundColor": "#fafafa",
                "marginBottom": "24px",
            },
        ),
        html.H2("About"),
        html.P(
            "This dashboard summarizes topic evolution in biomedical literature "
            "using clustering, trajectory analysis, and diffusion-based forecasting."
        ),
    ]

    return overview, movement_fig, cosine_fig, pca_fig, lineage_fig


if __name__ == "__main__":
    app.run(debug=True)