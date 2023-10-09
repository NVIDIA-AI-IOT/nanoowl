import time
import torch


class torch_timeit_sync():

    def __init__(self, name: str):
        self.name = name
        self.t0 = None

    def __enter__(self, *args, **kwargs):
        self.t0 = time.perf_counter_ns()

    def __exit__(self, *args, **kwargs):
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter_ns()
        dt = (t1 - self.t0) / 1e9
        print(f"{self.name} FPS: {round(1./dt, 3)}")