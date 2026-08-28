from __future__ import annotations

from datachecker.validation.raw_coverage_validator import RawCoverageValidator


def test_validate_counts_uses_normalized_kind_keys() -> None:
    validator = RawCoverageValidator(
        minimum_modality_counts={"audio": 5, "document": 2},
    )

    result = validator.validate_counts(
        counts={" Audio ": 5, " DOCUMENT ": 3},
    )

    assert result.is_valid
    assert result.counts == {"audio": 5, "document": 3}


def test_validate_counts_flags_normalized_shortfall() -> None:
    validator = RawCoverageValidator(
        minimum_modality_counts={"audio": 5},
    )

    result = validator.validate_counts(counts={" Audio ": 4})

    assert not result.is_valid
    assert "raw_modality_coverage_below_min:audio:4/5" in result.errors
    assert "raw_modality_missing:audio" not in result.errors


def test_validate_counts_flags_normalized_missing_kind() -> None:
    validator = RawCoverageValidator(
        minimum_modality_counts={"audio": 5, "video": 5},
    )

    result = validator.validate_counts(counts={" Audio ": 5})

    assert not result.is_valid
    assert "raw_modality_coverage_below_min:video:0/5" in result.errors
    assert "raw_modality_missing:video:target>5" in result.errors


def test_validate_counts_skips_zero_minima() -> None:
    validator = RawCoverageValidator(
        minimum_modality_counts={"page": 0, "audio": 5},
    )

    result = validator.validate_counts(counts={"audio": 6})

    assert result.is_valid


def test_validate_counts_rejects_negative_confirmed_counts() -> None:
    validator = RawCoverageValidator(minimum_modality_counts={"audio": 5})

    result = validator.validate_counts(counts={"audio": -3})

    assert not result.is_valid
