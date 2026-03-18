from pathlib import Path

import pandas as pd
from dash import Dash, html, dcc
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
DATA_PATH = PROJECT_ROOT / "data"

app = Dash(__name__)
app.title = "Diffusion Topic Dashboard"

future_csv = DATA_PATH / "future_topic_movement.csv"

if future_csv.exists():
    future_df = pd.read_csv(future_csv)

    fig = px.bar(
        future_df.sort_values("movement_norm", ascending=True),
        x="movement_norm",
        y="topic_label",
        orientation="h",
        title="Predicted Topic Movement",
    )
else:
    future_df = pd.DataFrame()
    fig = None


app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px"},
    children=[
        html.H1("Diffusion Topic Evolution Dashboard"),
        html.P(
            "This dashboard will later surface topic trajectories, lineage structure, "
            "and diffusion-based movement forecasts."
        ),
        html.H2("Current Status"),
        html.Ul(
            [
                html.Li(f"Outputs path: {OUTPUTS_PATH}"),
                html.Li(f"Data path: {DATA_PATH}"),
                html.Li(
                    f"Future movement loaded: {'yes' if not future_df.empty else 'no'}"
                ),
            ]
        ),
        html.H2("Preview"),
        dcc.Graph(figure=fig) if fig is not None else html.P("No preview data available yet."),
    ],
)


if __name__ == "__main__":
    app.run(debug=True)