"""Characterization tests for multimodal model initialization capabilities."""

from __future__ import annotations

import pytest
from torch import nn

from multimodal.model.initialization import freeze_model_components


class _Component(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(2, 2)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = _Component()
        self.fusion = _Component()
        self.projection = _Component()


def test_freeze_model_components_freezes_only_matching_prefixes() -> None:
    model = _Model()

    freeze_model_components(
        model=model,
        component_prefixes=("text_encoder",),
    )

    grad_state = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }

    assert grad_state
    for name, requires_grad in grad_state.items():
        if name.startswith("text_encoder"):
            assert requires_grad is False, name
        else:
            assert requires_grad is True, name


def test_freeze_model_components_without_matches_changes_nothing() -> None:
    model = _Model()
    before = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }

    freeze_model_components(
        model=model,
        component_prefixes=("vision_tower",),
    )

    after = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    assert after == before


@pytest.mark.parametrize("prefixes", [(), ("text_encoder", "fusion")])
def test_freeze_model_components_multiple_prefixes(
    prefixes: tuple[str, ...],
) -> None:
    model = _Model()

    freeze_model_components(model=model, component_prefixes=prefixes)

    for name, parameter in model.named_parameters():
        expected_frozen = any(name.startswith(p) for p in prefixes)
        assert parameter.requires_grad is not expected_frozen, name
