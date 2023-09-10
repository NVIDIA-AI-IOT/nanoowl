import torch.nn as nn


class ModuleRecorder(object):

    def __init__(self, module: nn.Module):
        self._module = module
        self._input = None
        self._output = None
        self._hook = None

    def _on_forward(self, module, input, output):
        self._input = input
        self._output = output

    def attach(self):
        if self._hook is not None:
            raise RuntimeError("Hook already attached.")
        self._hook = self._module.register_forward_hook(self._on_forward)

    def detach(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self, *args, **kwargs):
        self.attach()

    def __exit__(self, *args, **kwargs):
        self.detach()

    def get_input(self):
        return self._input
    
    def get_output(self):
        return self._output
