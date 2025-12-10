import json
import plotly.graph_objs as go
import plotly.offline as pyo
import os

objs_CPU = []
objs_GPU = []

def read_file() -> list:
    file_path = os.path.join(os.path.dirname(__file__), 'bench_result.json')
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["benchmarks"]

def parsing_json(list_obj):
    for obj in list_obj:
        if "CPU" in obj["name"]:
            objs_CPU.append({
                "name": obj["name"].split("/")[0],
                "number_elements": int(obj["name"].split("/")[1]),
                "real_time": float(obj["real_time"])
            })
        elif "GPU" in obj["name"]:
            objs_GPU.append({
                "name": obj["name"].split("/")[0],
                "number_elements": int(obj["name"].split("/")[1]),
                "real_time": float(obj["real_time"])
            })

def real_complexity_output_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "real_complexity_plot.html")

def plot():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[obj['number_elements'] for obj in objs_CPU],
        y=[obj['real_time'] for obj in objs_CPU],
        mode='lines+markers',
        name='Eigen Matrix Multiplication (CPU, float)',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=[obj['number_elements'] for obj in objs_GPU],
        y=[obj['real_time'] for obj in objs_GPU],
        mode='lines+markers',
        name='CUDA Matrix Multiplication (GPU, Naive, float)',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Real Complexity",
        xaxis=dict(
            title="N",
            type="log",
            tickvals=[16, 32, 64, 128, 256, 512, 1024],
            ticktext=["16", "32", "64", "128", "256", "512", "1024"]
        ),
        yaxis=dict(
            title="Time, ms",
            type="log",
            exponentformat="none"
        ),
        legend=dict(
            title="Benchmarks:",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2
        ),
        margin=dict(l=50, r=50, t=50, b=80),
    )

    pyo.plot(fig, filename=real_complexity_output_file(), auto_open=True)

def parse_data(list_obj):
    objs_CPU = []
    objs_GPU = []

    for obj in list_obj:
        name = obj["name"]
        n = int(name.split("/")[1])
        time = float(obj["real_time"])

        if "CPU" in name:
            objs_CPU.append((n, time))
        elif "GPU" in name:
            objs_GPU.append((n, time))

    objs_CPU.sort(key=lambda x: x[0])
    objs_GPU.sort(key=lambda x: x[0])

    speedup = []
    for (n_cpu, t_cpu), (n_gpu, t_gpu) in zip(objs_CPU, objs_GPU):
        if n_cpu == n_gpu:
            speedup.append({"N": n_cpu, "Speedup": t_cpu / t_gpu})

    return speedup

def speedup_output_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "speedup_plot.html")

def plot_speedup(speedup):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[d["N"] for d in speedup],
        y=[d["Speedup"] for d in speedup],
        mode="lines+markers",
        name="Speedup",
        line=dict(color="blue")
    ))

    fig.update_layout(
        title="Speedup: CUDA Matrix Multiplication (GPU, Naive, float) vs Eigen Matrix Multiplication (CPU, float)",
        xaxis=dict(
            title="N",
            type="log",
            tickvals=[16, 32, 64, 128, 256, 512, 1024],
            ticktext=["16", "32", "64", "128", "256", "512", "1024"]
        ),
        yaxis=dict(
            title="Speedup",
            type="log",
            exponentformat="none"
        ),
        margin=dict(l=50, r=50, t=50, b=80),
    )

    pyo.plot(fig, filename=speedup_output_file(), auto_open=True)

if __name__ == "__main__":
    data = read_file()
    speedup = parse_data(data)
    plot_speedup(speedup)
    parsing_json(data)
    plot()

