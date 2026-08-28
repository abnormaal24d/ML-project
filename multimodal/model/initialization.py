"""Deterministic initialization for a fully composed multimodal model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

INITIALIZATION_SCHEMA = "scratch_xavier_v1"


def freeze_model_components(
    *,
    model: nn.Module,
    component_prefixes: Sequence[str],
) -> None:
    """Freeze every parameter whose name starts with one component prefix.

    Parameters whose name matches none of the prefixes are left untouched.
    """

    prefixes = tuple(component_prefixes)

    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(False)


def initialize_model_from_scratch(
    *,
    model: nn.Module,
    seed: int,
) -> dict[str, object]:
    """Initialize every model parameter under one documented random seed."""

    initialized: set[int] = set()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for module in model.modules():
            _initialize_direct_parameters(
                module=module,
                initialized=initialized,
            )
        for parameter in model.parameters():
            if id(parameter) in initialized:
                continue
            _initialize_unowned_parameter(parameter)
            initialized.add(id(parameter))

    return {
        "seed": int(seed),
        "schema": INITIALIZATION_SCHEMA,
        "parameter_count": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _initialize_direct_parameters(
    *,
    module: nn.Module,
    initialized: set[int],
) -> None:
    for name, parameter in module.named_parameters(recurse=False):
        if id(parameter) in initialized:
            continue
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if name == "weight":
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    parameter[module.padding_idx].zero_()
        elif name.endswith("bias") or name == "bias":
            nn.init.zeros_(parameter)
        elif parameter.ndim >= 2:
            nn.init.xavier_uniform_(parameter)
        else:
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        initialized.add(id(parameter))


def _initialize_unowned_parameter(parameter: nn.Parameter) -> None:
    if parameter.ndim >= 2:
        nn.init.xavier_uniform_(parameter)
    else:
        nn.init.normal_(parameter, mean=0.0, std=0.02)
