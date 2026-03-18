from pathlib import Path
import pickle

import pandas as pd
from dash import Dash, html, dcc
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
DATA_PATH = PROJECT_ROOT / "data"

FUTURE_MOVEMENT_PATH = DATA_PATH / "future_topic_movement.csv"
TRAJ_PATH = DATA_PATH / "topic_trajectories.pkl"
LINEAGE_PATH = DATA_PATH / "hdbscan_lineage.csv"

app = Dash(__name__)
app.title = "Diffusion Topic Dashboard"


def load_future_df() -> pd.DataFrame:
    if FUTURE_MOVEMENT_PATH.exists():
        return pd.read_csv(FUTURE_MOVEMENT_PATH)
    return pd.DataFrame()


def artifact_status() -> dict:
    return {
        "future_movement": FUTURE_MOVEMENT_PATH.exists(),
        "trajectories": TRAJ_PATH.exists(),
        "lineage": LINEAGE_PATH.exists(),
    }


future_df = load_future_df()
status = artifact_status()

movement_fig = None
cosine_fig = None

if not future_df.empty:
    movement_fig = px.bar(
        future_df.sort_values("movement_norm", ascending=True),
        x="movement_norm",
        y="topic_label",
        orientation="h",
        title="Predicted 2023 Topic Movement (Neural Denoiser)",
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
        title="Predicted 2023 Stability Relative to 2022",
        labels={
            "cosine_similarity_to_latest": "Cosine similarity to 2022 state",
            "topic_label": "Topic",
        },
    )
    cosine_fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )


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
            "Interactive view of topic movement, topic stability, and later "
            "trajectory and lineage structure."
        ),

        html.H2("Current Status"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(f"Outputs path: {OUTPUTS_PATH}"),
                        html.Div(f"Data path: {DATA_PATH}"),
                    ],
                    style={"marginBottom": "12px"},
                ),
                status_badge("Future movement CSV", status["future_movement"]),
                status_badge("Topic trajectories PKL", status["trajectories"]),
                status_badge("HDBSCAN lineage CSV", status["lineage"]),
            ],
            style={
                "padding": "16px",
                "border": "1px solid #ddd",
                "borderRadius": "10px",
                "backgroundColor": "#fafafa",
                "marginBottom": "24px",
            },
        ),

        html.H2("Predicted Topic Movement"),
        dcc.Graph(figure=movement_fig) if movement_fig is not None else html.P(
            "No future movement data available yet."
        ),

        html.H2("Predicted Topic Stability"),
        dcc.Graph(figure=cosine_fig) if cosine_fig is not None else html.P(
            "No cosine similarity data available yet."
        ),

        html.H2("Coming Next"),
        html.Ul(
            [
                html.Li("Interactive PCA topic trajectories"),
                html.Li("Interactive HDBSCAN lineage evolution"),
            ]
        ),
    ],
)


if __name__ == "__main__":
    app.run(debug=True)