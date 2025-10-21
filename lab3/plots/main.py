import json
import plotly.graph_objs as go
import plotly.offline as pyo
import os

def read_file() -> list:
    file_path = os.path.join(os.path.dirname(__file__), 'plots')
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["benchmarks"]

def parse_benchmarks(data):
    eigen = []
    cuda_naive = []
    cuda_shared = []

    for obj in data:
        name = obj["name"]
        n = int(name.split("/")[-1])
        t = float(obj["real_time"])

        if "Eigen" in name:
            eigen.append((n, t))
        elif "Naive" in name:
            cuda_naive.append((n, t))
        elif "Shared" in name:
            cuda_shared.append((n, t))

    eigen.sort(key=lambda x: x[0])
    cuda_naive.sort(key=lambda x: x[0])
    cuda_shared.sort(key=lambda x: x[0])

    return eigen, cuda_naive, cuda_shared

def plot_real_complexity(eigen, cuda_naive, cuda_shared):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[n for n, _ in eigen],
        y=[t for _, t in eigen],
        mode='lines+markers',
        name='Eigen Matrix (CPU)',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=[n for n, _ in cuda_naive],
        y=[t for _, t in cuda_naive],
        mode='lines+markers',
        name='CUDA Naive',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=[n for n, _ in cuda_shared],
        y=[t for _, t in cuda_shared],
        mode='lines+markers',
        name='CUDA Shared',
        line=dict(color='green')
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

def compute_speedup(naive, shared):
    speedup = []
    for (n1, t1), (n2, t2) in zip(naive, shared):
        if n1 == n2:
            speedup.append((n1, t1 / t2))
    return speedup

def plot_speedup(speedup):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[n for n, _ in speedup],
        y=[s for _, s in speedup],
        mode='lines+markers',
        name='Speedup (old/new)',
        line=dict(color='purple')
    ))

    fig.update_layout(
        title="Speedup: New operator* (Shared) vs Old operator* (Naive)",
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
    eigen, cuda_naive, cuda_shared = parse_benchmarks(data)

    plot_real_complexity(eigen, cuda_naive, cuda_shared)

    speedup = compute_speedup(cuda_naive, cuda_shared)
    plot_speedup(speedup)
