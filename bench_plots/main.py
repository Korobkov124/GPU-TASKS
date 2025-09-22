import json
import matplotlib.pyplot as plt
import os
import sys

objs_GPUFull = list()
objs_GPUCore = list()
objs_CPU = list()


def read_file() -> list:
    file_path = os.path.join(os.path.dirname(__file__), 'bench_result.json')
    
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["benchmarks"]


def plot():
    plt.figure(figsize=(15, 12))
    
    cpu_x = [obj['number_elements'] for obj in objs_CPU]
    cpu_y = [obj['real_time'] / 1000000 for obj in objs_CPU]
    plt.plot(cpu_x, cpu_y, marker='o', label="CPU", linewidth=2)
    

    gpu_x = [obj['number_elements'] for obj in objs_GPUFull]
    gpu_y = [obj['real_time'] / 1000000 for obj in objs_GPUFull]
    plt.plot(gpu_x, gpu_y, marker='s', label="GPU", linewidth=2)

    plt.title("ms/number_elements")
    plt.xlabel("Number_elements")
    plt.ylabel("Execution_time(ms)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig(path_output_file(), dpi=300, bbox_inches='tight')
    plt.show()


def path_output_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "bench_plot.png")
    return output_file


def parsing_json(list_obj):
    for obj in list_obj:
        if "CPU" in obj["name"]:
            obj_CPU = {"name": (obj["name"].split("/"))[0], "number_elements": (obj["name"].split("/"))[1], "real_time": obj["real_time"]}
            objs_CPU.append(obj_CPU)
        elif "GPU" in obj["name"]:
            obj_GPU = {"name": (obj["name"].split("/"))[0], "number_elements": (obj["name"].split("/"))[1], "real_time": obj["real_time"]}
            objs_GPUFull.append(obj_GPU)


if __name__ == '__main__':
    list_obj = read_file()
    parsing_json(list_obj)
    plot()