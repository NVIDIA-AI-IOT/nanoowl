import torch
import torch.nn as nn
import time

class ModuleProfiler(object):

    def __init__(self, module: nn.Module):
        self._module = module
        self._forward_pre_hook = None
        self._forward_hook = None
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def _on_pre_forward(self, module, input):
        self._start.record()
        
    def _on_forward(self, module, input, output):
        self._end.record()

    def attach(self):
        if self._forward_hook is not None:
            raise RuntimeError("Hook already attached.")
        if self._forward_pre_hook is not None:
            raise RuntimeError("Hook already attached.")
        self._forward_pre_hook = self._module.register_forward_pre_hook(self._on_pre_forward)
        self._forward_hook = self._module.register_forward_hook(self._on_forward)

    def detach(self):
        if self._forward_hook is not None:
            self._forward_hook.remove()
            self._forward_hook = None
        if self._forward_pre_hook is not None:
            self._forward_pre_hook.remove()
            self._forward_pre_hook = None

    def __enter__(self, *args, **kwargs):
        self.attach()

    def __exit__(self, *args, **kwargs):
        self.detach()

    def get_elapsed_time(self):
        self._end.wait()
        return self._start.elapsed_time(self._end)