"""Consolidated named-entities privacy detectors."""

from preprocessing.privacy.inspection.context_phrase_detector import (
    ContextPhraseDetector,
    phrase,
)
from preprocessing.privacy.inspection.detector import TextDetector
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.pattern_detector import (
    PatternDetector,
    PatternSpec,
    compile_pattern,
)


def _valid_coordinates(value: str) -> bool:
    try:
        left, right = value.replace(";", ",").split(",", maxsplit=1)
        latitude, longitude = float(left.strip()), float(right.strip())
    except ValueError:
        return False
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def build_geographic_location_detector() -> PatternDetector:
    return PatternDetector(
        name="geographic_coordinates",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.GEOGRAPHIC_LOCATION,
                pattern=compile_pattern(
                    r"(?<![\d.])-?\d{1,2}(?:\.\d{4,})\s*[,;]\s*"
                    r"-?\d{1,3}(?:\.\d{4,})(?![\d.])"
                ),
                confidence=0.97,
                validator=_valid_coordinates,
            ),
        ),
    )


def build_organization_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="organization_context",
        version="1.0.0",
        finding_type=FindingType.ORGANIZATION,
        phrases=(
            phrase(
                r"(?:werkgever|employer|organisation|organization|bedrijf)"
                r"\s*[:=-]\s*[^\n,;]{2,80}",
                0.75,
            ),
        ),
    )


def build_person_name_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="person_name_context",
        version="1.0.0",
        finding_type=FindingType.PERSON_NAME,
        phrases=(
            phrase(
                r"(?:naam|name|nom|patient|pati[eë]nt|client)\s*[:=-]\s*"
                r"[A-ZÀ-ÖØ-Ý][\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[A-ZÀ-ÖØ-Ý]"
                r"[\wÀ-ÖØ-öø-ÿ'’-]+){1,3}",
                0.91,
            ),
            phrase(
                r"(?:mr|mrs|ms|dr|meneer|mevrouw|monsieur|madame)\.?\s+"
                r"[A-ZÀ-ÖØ-Ý][\wÀ-ÖØ-öø-ÿ'’-]+",
                0.84,
            ),
        ),
    )


def build_postal_address_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="postal_address_context",
        version="1.0.0",
        finding_type=FindingType.POSTAL_ADDRESS,
        phrases=(
            phrase(
                r"\b[A-ZÀ-ÖØ-Ý][\wÀ-ÖØ-öø-ÿ'’-]+(?:straat|laan|weg|steenweg|"
                r"plein|kaai|dreef|markt|rue|avenue|boulevard|chauss[eé]e|straat)"
                r"\s+\d{1,5}[A-Za-z]?(?:\s*(?:bus|box|bte)\s*\w+)?"
                r"(?:,?\s*\d{4,5}\s+[A-ZÀ-ÖØ-Ý][\wÀ-ÖØ-öø-ÿ'’ -]+)?",
                0.9,
            ),
            phrase(r"(?:adres|address|adresse)\s*[:=-]\s*[^\n]{8,100}", 0.88),
        ),
    )


def build_named_entity_detectors() -> tuple[TextDetector, ...]:
    """Construct the default named-entity detector family in stable order."""

    return (
        build_person_name_detector(),
        build_postal_address_detector(),
        build_geographic_location_detector(),
        build_organization_detector(),
    )
