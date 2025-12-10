# vector_plots.py
import json
import os
import math
import plotly.graph_objs as go
import plotly.offline as pyo
from typing import List, Tuple, Dict

PLOTS_JSON = "plots.json"

def read_file() -> dict:
    path = os.path.join(os.path.dirname(__file__), PLOTS_JSON)
    with open(path, "r") as f:
        return json.load(f)

def parse_benchmarks(data: dict):
    benchmarks = data.get("benchmarks", [])
    context = data.get("context", {})
    time_unit = context.get("time_unit", None)
    if time_unit is None:
        print("Warning: time_unit not found in JSON context — assuming 'ms' (milliseconds).")
        time_unit = "ms"

    unit_to_ms = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    if time_unit not in unit_to_ms:
        print(f"Unknown time_unit '{time_unit}', treating as milliseconds.")
    factor = unit_to_ms.get(time_unit, 1.0)

    nobr = []
    br = []
    other = []

    for obj in benchmarks:
        name = obj.get("name", "")
        
        if "real_time" not in obj:
            continue
        raw_time = float(obj["real_time"])
        time_ms = raw_time * factor

        
        n = None
        if "/" in name:
            try:
                token = name.split("/")[-1]
                n = int(token)
            except Exception:
                n = None
        if n is None:
            import re
            m = re.search(r'(\d{1,10})', name)
            if m:
                n = int(m.group(1))

        lname = name.lower()
        is_nobr = ("nobr" in lname) or ("vecred_nobr" in lname) or ("kernel_vecred_nobr" in lname) or ("nobr" in name)
        is_br   = ("br" in lname and not "nobr" in lname) or ("vecred_br" in lname) or ("kernel_vecred_br" in lname)

        entry = (n if n is not None else -1, time_ms, name)
        if is_nobr:
            nobr.append(entry)
        elif is_br:
            br.append(entry)
        else:
            other.append(entry)

    nobr = [(n, t) for n, t, nm in sorted(nobr, key=lambda x: (x[0] if x[0]>=0 else math.inf))]
    br   = [(n, t) for n, t, nm in sorted(br,   key=lambda x: (x[0] if x[0]>=0 else math.inf))]

    if other:
        print("Note: some benchmark items were not classified as nobr/br (listing up to 5):")
        for e in other[:5]:
            print("  ", e)

    return nobr, br

def align_pairs(a: List[Tuple[int, float]], b: List[Tuple[int, float]]):

    da = {n: t for n, t in a}
    db = {n: t for n, t in b}
    common_ns = sorted(set(da.keys()) & set(db.keys()))
    pairs = [(n, da[n], db[n]) for n in common_ns]
    return pairs

def plot_real_complexity(nobr, br, outname="real_complexity_plot.html"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[n for n, _ in nobr],
        y=[t for _, t in nobr],
        mode='lines+markers',
        name='kernel_vecred_nobr',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=[n for n, _ in br],
        y=[t for _, t in br],
        mode='lines+markers',
        name='kernel_vecred_br',
        line=dict(color='blue')
    ))

    max_n = max([n for n,_ in (nobr+br)]) if (nobr+br) else 1
    tickvals = []
    val = 8
    while val <= max_n*2:
        tickvals.append(val)
        val *= 2

    fig.update_layout(
        title="Real Computational Complexity of vector reduction kernels",
        xaxis=dict(
            title="Vector size (n)",
            type="log",
            tickvals=tickvals,
            ticktext=[str(v) for v in tickvals]
        ),
        yaxis=dict(
            title="Execution Time (ms)",
            type="log",
            exponentformat="none"
        ),
        legend=dict(title="Kernel", orientation="h", x=0.5, xanchor="center", y=-0.2),
        margin=dict(l=50, r=50, t=60, b=80),
    )

    pyo.plot(fig, filename=outname, auto_open=True)
    print(f"Wrote {outname}")

def plot_speedup(pairs, outname="speedup_plot.html"):
    ns = [n for n,_,_ in pairs]
    speedups = [ (ta / tb if tb > 0 else float('nan')) for _, ta, tb in pairs ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns,
        y=speedups,
        mode='lines+markers',
        name='Speedup = time_nobr / time_br',
        line=dict(color='purple')
    ))

    max_n = max(ns) if ns else 1
    tickvals = []
    val = 8
    while val <= max_n*2:
        tickvals.append(val)
        val *= 2

    fig.update_layout(
        title="Speedup: kernel_vecred_br vs kernel_vecred_nobr",
        xaxis=dict(title="Vector size (n)", type="log", tickvals=tickvals, ticktext=[str(v) for v in tickvals]),
        yaxis=dict(title="Speedup (×)", type="linear"),
        margin=dict(l=50, r=50, t=60, b=80),
    )

    pyo.plot(fig, filename=outname, auto_open=True)
    print(f"Wrote {outname}")

if __name__ == "__main__":
    data = read_file()
    nobr, br = parse_benchmarks(data)

    if not nobr:
        print("ERROR: no 'nobr' benchmarks found.")
    if not br:
        print("ERROR: no 'br' benchmarks found.")

    plot_real_complexity(nobr, br, outname="vector_real_complexity_plot.html")

    pairs = align_pairs(nobr, br)
    if not pairs:
        print("WARNING: no matching sizes between nobr and br — speedup plot will be empty.")
    plot_speedup(pairs, outname="vector_speedup_plot.html")
