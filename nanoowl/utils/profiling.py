import torch
import time
import functools
import statistics
from collections import OrderedDict


class Profiler:

    active_profilers = set()

    def __init__(self):
        self.stack = []
        self.elapsed_times = OrderedDict()

    def __enter__(self, *args, **kwargs):
        Profiler.active_profilers.add(self)
        return self

    def __exit__(self, *args, **kwargs):
        Profiler.active_profilers.remove(self)

    def current_namespace(self):
        return ".".join(self.stack)

    def add_elapsed_time(self, timing_ms):
        namespace = self.current_namespace()
        if namespace not in self.elapsed_times:
            self.elapsed_times[namespace] = [timing_ms]
        else:
            self.elapsed_times[namespace].append(timing_ms)

    def mean_elapsed_times(self):
        times = OrderedDict()
        for k, v in self.elapsed_times.items():
            times[k] = statistics.mean(v)
        return times

    def median_elapsed_times(self):
        times = OrderedDict()
        for k, v in self.elapsed_times.items():
            times[k] = statistics.median(v)
        return times

    def print_mean_elapsed_times_ms(self):
        for k, v in self.mean_elapsed_times().items():
            print(f"{k}: {round(v, 3) / 1e6}")

    def print_median_elapsed_times_ms(self):
        for k, v in self.median_elapsed_times().items():
            print(f"{k}: {round(v, 3) / 1e6}")

    def clear(self):
        self.elapsed_times = OrderedDict()


class Timer:
    
    def __init__(self, scope: str):
        self.scope = scope
        self._t0 = None
        self._t1 = None

    def is_active(self):
        return len(Profiler.active_profilers) > 0
    
    def __enter__(self, *args, **kwargs):
        if not self.is_active():
            return self
        for profiler in Profiler.active_profilers:
            profiler.stack.append(self.scope)
        torch.cuda.current_stream().synchronize()
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *args, **kwargs):
        if not self.is_active():
            return self
        torch.cuda.current_stream().synchronize()
        self._t1 = time.perf_counter_ns()
        elapsed_time = self.get_elapsed_time_ns()
        for profiler in Profiler.active_profilers:
            profiler.add_elapsed_time(elapsed_time)
            profiler.stack.pop()
        return self

    def get_elapsed_time_ns(self):
        return (self._t1 - self._t0)


    def __call__(self, fn):

        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            with self:
                output = fn(*args, **kwargs)
            return output
        
        return _wrapper


def use_timer(fn):
    return Timer(fn.__qualname__)(fn)


def capture_timings():
    return Profiler()
