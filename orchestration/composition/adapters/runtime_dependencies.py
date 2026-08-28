"""Concrete runtime primitives used by application composition."""

import datetime
import uuid
from dataclasses import dataclass


class SystemClock:
    """Return timezone-aware UTC timestamps."""

    def now(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)


class UuidIdGenerator:
    """Generate UUID4 hexadecimal identifiers."""

    def generate(self) -> str:
        return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class RuntimePrimitives:
    """Shared clock and identifier generator for one runtime object graph."""

    clock: SystemClock
    id_generator: UuidIdGenerator


def build_runtime_primitives() -> RuntimePrimitives:
    """Construct production runtime primitives inside composition."""

    return RuntimePrimitives(
        clock=SystemClock(),
        id_generator=UuidIdGenerator(),
    )


__all__ = [
    "RuntimePrimitives",
    "SystemClock",
    "UuidIdGenerator",
    "build_runtime_primitives",
]
