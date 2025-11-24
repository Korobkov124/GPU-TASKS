import json
import plotly.graph_objs as go
import plotly.offline as pyo
import os

def read_file() -> list:
    file_path = os.path.join(os.path.dirname(__file__), 'plots.json')
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["benchmarks"]

def parse_benchmarks(data):
    shared = []
    wmma = []

    for obj in data:
        name = obj["name"]
        n = int(name.split("/")[-1])
        t = float(obj["real_time"])

        if "Shared" in name:
            shared.append((n, t))
        elif "WMMA" in name:
            wmma.append((n, t))

    shared.sort(key=lambda x: x[0])
    wmma.sort(key=lambda x: x[0])

    return shared, wmma

def plot_real_complexity(shared, wmma):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[n for n, _ in shared],
        y=[t for _, t in shared],
        mode='lines+markers',
        name='Old operator* (Shared)',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=[n for n, _ in wmma],
        y=[t for _, t in wmma],
        mode='lines+markers',
        name='New operator* (WMMA)',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title="Real Computational Complexity of Matrix Multiplication",
        xaxis=dict(
            title="Matrix size (N)",
            type="log",
            tickvals=[16, 32, 64, 128, 256, 512, 1024],
            ticktext=["16", "32", "64", "128", "256", "512", "1024"]
        ),
        yaxis=dict(
            title="Execution Time (ms)",
            type="log",
            exponentformat="none"
        ),
        legend=dict(
            title="Implementation:",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2
        ),
        margin=dict(l=50, r=50, t=50, b=80),
    )

    output_path = os.path.join(os.path.dirname(__file__), "real_complexity_plot.html")
    pyo.plot(fig, filename=output_path, auto_open=True)

def compute_speedup(shared, wmma):
    speedup = []
    for (n1, t1), (n2, t2) in zip(shared, wmma):
        if n1 == n2:
            speedup.append((n1, t1 / t2))
    return speedup

def plot_speedup(speedup):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[n for n, _ in speedup],
        y=[s for _, s in speedup],
        mode='lines+markers',
        name='Speedup (WMMA / Shared)',
        line=dict(color='purple')
    ))

    fig.update_layout(
        title="Speedup: New operator* (WMMA) vs Old operator* (Shared)",
        xaxis=dict(
            title="Matrix size (N)",
            type="log",
            tickvals=[16, 32, 64, 128, 256, 512, 1024],
            ticktext=["16", "32", "64", "128", "256", "512", "1024"]
        ),
        yaxis=dict(
            title="Speedup (×)",
            type="log",
            exponentformat="none"
        ),
        margin=dict(l=50, r=50, t=50, b=80),
    )

    output_path = os.path.join(os.path.dirname(__file__), "speedup_plot.html")
    pyo.plot(fig, filename=output_path, auto_open=True)

if __name__ == "__main__":
    data = read_file()
    shared, wmma = parse_benchmarks(data)

    plot_real_complexity(shared, wmma)

    speedup = compute_speedup(shared, wmma)
    plot_speedup(speedup)
