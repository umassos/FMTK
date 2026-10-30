"""Per-call overhead of TaskGraph.forward against calling the same components directly."""
import time

import torch
import torch.nn as nn

from fmtk.task_graph import TaskGraph

N = 20_000


def time_per_call(fn, x) -> float:
    for _ in range(1_000):
        fn(x)
    start = time.perf_counter()
    for _ in range(N):
        fn(x)
    return (time.perf_counter() - start) / N * 1e6


def main():
    parts = [nn.Linear(16, 16).eval() for _ in range(3)]
    g = TaskGraph()
    for i, p in enumerate(parts):
        g.add_component(f"c{i}", p)
    g.register_task("t", ["c0", "c1", "c2"])
    x = torch.randn(1, 16)

    def direct(z):
        with torch.inference_mode():
            for p in parts:
                z = p.forward(z)
        return z

    direct_us = time_per_call(direct, x)
    graph_us = time_per_call(lambda z: g.forward(z, "t"), x)
    print(f"direct {direct_us:.2f} us/call, graph {graph_us:.2f} us/call, overhead {graph_us - direct_us:.2f} us/call")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
