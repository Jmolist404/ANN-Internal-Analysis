# Processing activation data

import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any, Tuple


@dataclass
class HookPack:
    """Stores captured tensors in CPU to avoid GPU memory growth."""
    activations: Dict[str, List[torch.Tensor]]
    pre_activations: Dict[str, List[torch.Tensor]]
    handles: List[Any]

    def clear(self) -> None:
        self.activations.clear()
        self.pre_activations.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def _to_cpu_detached(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu")


def attach_activation_hooks(
    model: nn.Module,
    layer_names: List[str],
    capture_pre: bool = True,
    capture_post: bool = True,
) -> HookPack:
    """
    Attaches hooks to layers by their module name (as returned by model.named_modules()).

    capture_pre: forward_pre_hook -> inputs to the layer (pre-activation / internal signal)
    capture_post: forward_hook -> outputs of the layer (activation)
    """
    name_to_module = dict(model.named_modules())

    activations: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    pre_activations: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    handles: List[Any] = []

    def make_fwd_hook(layer_key: str) -> Callable:
        def hook(_m: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            # out could be tuple for some modules; handle common cases
            if isinstance(out, (tuple, list)):
                out_t = out[0]
            else:
                out_t = out
            activations[layer_key].append(_to_cpu_detached(out_t))
        return hook

    def make_pre_hook(layer_key: str) -> Callable:
        def hook(_m: nn.Module, inp: Tuple[torch.Tensor, ...]):
            # inp is a tuple; usually inp[0] is the tensor
            x = inp[0] if isinstance(inp, (tuple, list)) else inp
            if isinstance(x, (tuple, list)):
                x = x[0]
            if torch.is_tensor(x):
                pre_activations[layer_key].append(_to_cpu_detached(x))
        return hook

    for ln in layer_names:
        if ln not in name_to_module:
            raise ValueError(
                f"Layer '{ln}' not found in model.named_modules(). "
                f"Available examples: {list(name_to_module.keys())[:30]} ..."
            )

        module = name_to_module[ln]

        if capture_pre:
            handles.append(module.register_forward_pre_hook(make_pre_hook(ln)))
        if capture_post:
            handles.append(module.register_forward_hook(make_fwd_hook(ln)))

    return HookPack(activations=activations, pre_activations=pre_activations, handles=handles)


def concat_hooked_tensors(store: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Converts {layer: [batch1, batch2, ...]} -> {layer: full_tensor}
    """
    out: Dict[str, torch.Tensor] = {}
    for k, batches in store.items():
        if len(batches) == 0:
            out[k] = torch.empty(0)
        else:
            out[k] = torch.cat(batches, dim=0)
    return out

def list_layer_names(model, include_root: bool = False):
    names = [n for n, _ in model.named_modules()]
    if not include_root:
        names = [n for n in names if n != ""]
    return names
