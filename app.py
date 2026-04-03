from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objs as go
import json
import os

app = Dash(__name__)

# global state
automation_running = False

def load_metrics():
    try:
        with open("metrics.json", "r") as f:
            return json.load(f)
    except:
        return {"fps": 0, "success": 0, "energy": 0}

# layout
app.layout = html.Div([

    html.H1("🤖 Smart Robotic Dashboard"),

    dcc.Tabs([

        # ================= CONTROL TAB =================
        dcc.Tab(label="Control Panel", children=[

            html.Div([
                html.Button("▶ Start Automation", id="start-btn"),
                html.Button("⏹ Stop", id="stop-btn"),
            ]),

            html.H3(id="robot-status"),
            html.H3(id="camera-status"),

        ]),

        # ================= METRICS TAB =================
        dcc.Tab(label="Evaluation Metrics", children=[

            html.Div([
                html.Div(id="fps"),
                html.Div(id="success"),
                html.Div(id="energy"),
            ]),

            dcc.Graph(id="live-graph"),

        ])
    ]),

    dcc.Interval(id="interval", interval=1000, n_intervals=0)

])
@app.callback(
    Output("robot-status", "children"),
    Output("camera-status", "children"),
    Input("start-btn", "n_clicks"),
    Input("stop-btn", "n_clicks"),
)
def control(start, stop):

    if start and (not stop or start > stop):
        with open("control.json", "w") as f:
            json.dump({"run": True}, f)

        return "✅ Robot Connected (Dummy)", "📷 Camera Started"

    if stop and (not start or stop > start):
        with open("control.json", "w") as f:
            json.dump({"run": False}, f)

        return "⛔ Stopped", "Camera Off"

    return "Idle", "Camera idle"
fps_history = []
success_history = []
time_history = []

@app.callback(
    Output("fps", "children"),
    Output("success", "children"),
    Output("energy", "children"),
    Output("live-graph", "figure"),
    Input("interval", "n_intervals")
)
def update(n):

    data = load_metrics()

    fps_history.append(data["fps"])
    success_history.append(data["success"] * 100)
    time_history.append(n)

    fps_history[:] = fps_history[-50:]
    success_history[:] = success_history[-50:]
    time_history[:] = time_history[-50:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_history,
        y=fps_history,
        mode='lines+markers',
        name='FPS'
    ))

    fig.add_trace(go.Scatter(
        x=time_history,
        y=success_history,
        mode='lines+markers',
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