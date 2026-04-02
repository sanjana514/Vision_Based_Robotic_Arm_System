from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from utils import load_metrics
import time

app = Dash(__name__)

# store history for graph
fps_history = []
success_history = []
time_history = []

app.layout = html.Div([
    
    html.H1("🤖 Robotic Arm Dashboard", style={"textAlign": "center"}),

    # 🔷 Metrics Row
    html.Div([
        html.Div(id="fps", className="card"),
        html.Div(id="success", className="card"),
        html.Div(id="energy", className="card"),
    ], style={"display": "flex", "justifyContent": "space-around"}),

    # 🔷 Graph
    dcc.Graph(id="live-graph"),

    # 🔷 Auto refresh
    dcc.Interval(
        id="interval",
        interval=1000,  # 1 sec
        n_intervals=0
    )
])


# 🔁 Callback
@app.callback(
    [
        Output("fps", "children"),
        Output("success", "children"),
        Output("energy", "children"),
        Output("live-graph", "figure"),
    ],
    [Input("interval", "n_intervals")]
)
def update_dashboard(n):

    data = load_metrics()

    # update history
    fps_history.append(data["fps"])
    success_history.append(data["success"] * 100)
    time_history.append(n)

    # limit history (last 50 points)
    fps_history[:] = fps_history[-50:]
    success_history[:] = success_history[-50:]
    time_history[:] = time_history[-50:]

    # graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_history,
        y=fps_history,
        mode='lines',
        name='FPS'
    ))

    fig.add_trace(go.Scatter(
        x=time_history,
        y=success_history,
        mode='lines',
        name='Success %'
    ))

    return (
        f"FPS: {data['fps']:.2f}",
        f"Success: {data['success']*100:.1f}%",
        f"Energy: {data['energy']:.2f}",
        fig
    )


if __name__ == "__main__":
    app.run(debug=True)