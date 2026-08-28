"""Registered audio augmentation operations and canonical transform settings."""

from __future__ import annotations

from dataclasses import dataclass

_SUPPORTED_OPERATIONS = frozenset(
    {
        "gain_shift",
        "noise_injection",
        "trim",
        "speed_perturbation",
        "sample_rate_normalization",
        "channel_conversion",
    }
)


@dataclass(frozen=True, slots=True)
class AudioTransformParameters:
    """Fully resolved parameters for one deterministic audio transform."""

    operations: tuple[str, ...]
    target_sample_rate: int
    target_channels: int
    gain_db: float
    noise_std_fraction: float
    trim_silence_threshold_dbfs: float
    trim_padding_seconds: float
    speed_factor: float
    noise_seed: int

    @property
    def normalize_sample_rate(self) -> bool:
        return "sample_rate_normalization" in self.operations

    @property
    def convert_channels(self) -> bool:
        return "channel_conversion" in self.operations

    @property
    def trim_silence(self) -> bool:
        return "trim" in self.operations

    @property
    def perturb_speed(self) -> bool:
        return "speed_perturbation" in self.operations

    @property
    def shift_gain(self) -> bool:
        return "gain_shift" in self.operations

    @property
    def inject_noise(self) -> bool:
        return "noise_injection" in self.operations


def resolve_audio_operations(names: tuple[str, ...]) -> tuple[str, ...]:
    """Validate configured names and return a duplicate-free ordered tuple."""

    unknown = set(names) - _SUPPORTED_OPERATIONS
    if unknown:
        raise ValueError(
            f"unknown audio augmentation operations: {sorted(unknown)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("audio augmentation operations must be unique")
    return names
