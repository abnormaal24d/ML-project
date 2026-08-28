"""Robots access-action projection tests (allow/block/defer)."""

from __future__ import annotations

import pytest

from crawler.governance.robots.robots_check_result import (
    RobotsAccessAction,
    RobotsCheckResult,
    RobotsConfidence,
    RobotsDecision,
)


def _result(
    *,
    decision: RobotsDecision,
    confidence: RobotsConfidence | None = None,
) -> RobotsCheckResult:
    return RobotsCheckResult(
        robots_url="https://example.test/robots.txt",
        decision=decision,
        reason="test",
        source="test",
        is_authoritative=True,
        confidence=confidence,
    )


def _project(
    result: RobotsCheckResult,
    *,
    on_weak_unknown: str = "block",
    on_transient_unknown: str = "defer",
    on_hostile_unknown: str = "block",
) -> RobotsAccessAction:
    return result.to_access_action(
        on_weak_unknown=on_weak_unknown,
        on_transient_unknown=on_transient_unknown,
        on_hostile_unknown=on_hostile_unknown,
    )


def test_action_enum_values() -> None:
    assert RobotsAccessAction.ALLOW == "allow"
    assert RobotsAccessAction.BLOCK == "block"
    assert RobotsAccessAction.DEFER == "defer"


def test_allowed_projects_to_allow_ignoring_unknown_policy() -> None:
    result = _result(
        decision=RobotsDecision.ALLOWED,
        confidence=RobotsConfidence.AUTHORITATIVE_ALLOW,
    )
    assert _project(result) == RobotsAccessAction.ALLOW


def test_disallowed_projects_to_block() -> None:
    result = _result(
        decision=RobotsDecision.DISALLOWED,
        confidence=RobotsConfidence.AUTHORITATIVE_DENY,
    )
    assert _project(result) == RobotsAccessAction.BLOCK


@pytest.mark.parametrize(
    ("confidence", "policy_key"),
    [
        (RobotsConfidence.HOSTILE_UNKNOWN, "on_hostile_unknown"),
        (RobotsConfidence.TRANSIENT_UNKNOWN, "on_transient_unknown"),
        (RobotsConfidence.WEAK_UNKNOWN, "on_weak_unknown"),
        (None, "on_weak_unknown"),
    ],
)
def test_unknown_uses_confidence_specific_policy(
    confidence: RobotsConfidence | None,
    policy_key: str,
) -> None:
    """Each UNKNOWN confidence is governed by its own policy setting."""

    result = _result(decision=RobotsDecision.UNKNOWN, confidence=confidence)
    policy = {
        "on_weak_unknown": "allow",
        "on_transient_unknown": "allow",
        "on_hostile_unknown": "allow",
    }
    policy[policy_key] = "block"
    action = result.to_access_action(**policy)

    assert action == RobotsAccessAction.BLOCK


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("allow", RobotsAccessAction.ALLOW),
        ("block", RobotsAccessAction.BLOCK),
        ("defer", RobotsAccessAction.DEFER),
    ],
)
def test_action_constructible_from_literal_string(
    value: str,
    expected: RobotsAccessAction,
) -> None:
    assert RobotsAccessAction(value) is expected
