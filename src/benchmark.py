import platform
import time
import os
import subprocess


try:
    import psutil
except ImportError:
    psutil = None


def get_hardware_info():
    info = {
        "os": platform.system() + " " + platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores": os.cpu_count(),
    }

    if psutil:
        info["ram_gb"] = round(
            psutil.virtual_memory().total / (1024 ** 3), 2
        )
    else:
        info["ram_gb"] = "Not available"

    return info

def get_gpu_info():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return "No NVIDIA GPU detected"


def benchmark_function(function):
    start = time.perf_counter()

    result = function()

    end = time.perf_counter()

    execution_time = end - start

    return result, execution_time


if __name__ == "__main__":

    hardware = get_hardware_info()
    hardware["gpu"] = get_gpu_info()

    print("Hardware Information")
    print("--------------------")

    for key, value in hardware.items():
        print(f"{key}: {value}")

    print("\nBenchmark Test")
    print("--------------")

    def test_operation():
        total = 0

        for i in range(1_000_000):
            total += i

        return total

    _, execution_time = benchmark_function(test_operation)

    print(f"Execution Time: {execution_time:.6f} seconds")