"""Consolidated sensitive-information privacy detectors."""

from preprocessing.privacy.inspection.context_phrase_detector import (
    ContextPhraseDetector,
    phrase,
)
from preprocessing.privacy.inspection.detector import TextDetector
from preprocessing.privacy.inspection.finding_type import FindingType


def build_criminal_information_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="criminal_information_context",
        version="1.0.0",
        finding_type=FindingType.CRIMINAL_INFORMATION,
        phrases=(
            phrase(
                r"\b(?:strafblad|criminal record|veroordeeld|convicted|verdachte|suspect|arrested|gearresteerd)\b",
                0.78,
            ),
        ),
    )


def build_financial_information_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="financial_information_context",
        version="1.0.0",
        finding_type=FindingType.FINANCIAL_INFORMATION,
        phrases=(
            phrase(
                r"\b(?:salary|salaris|loon|income|inkomen|debt|schuld|balance|saldo|tax|belasting|bankrekening)\b",
                0.78,
            ),
        ),
    )


def build_health_information_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="health_information_context",
        version="1.0.0",
        finding_type=FindingType.HEALTH_INFORMATION,
        phrases=(
            phrase(
                r"\b(?:diagnos(?:e|is)|diabetes|kanker|cancer|hiv|medic(?:atie|ation)|patient|pati[eë]nt|ziekte|disease|treatment|behandeling)\b",
                0.78,
            ),
        ),
    )


def build_minor_information_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="minor_information_context",
        version="1.0.0",
        finding_type=FindingType.MINOR_INFORMATION,
        phrases=(
            phrase(
                r"\b(?:minderjarige|minor|child|kind|leerling|pupil)\b.{0,40}\b(?:jaar|years? old|age|leeftijd)\b",
                0.78,
            ),
        ),
    )


def build_political_information_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="political_information_context",
        version="1.0.0",
        finding_type=FindingType.POLITICAL_INFORMATION,
        phrases=(
            phrase(
                r"\b(?:politieke overtuiging|political affiliation|party member|partijlid|stemgedrag|voting preference)\b",
                0.78,
            ),
        ),
    )


def build_religious_information_detector() -> ContextPhraseDetector:
    return ContextPhraseDetector(
        name="religious_information_context",
        version="1.0.0",
        finding_type=FindingType.RELIGIOUS_INFORMATION,
        phrases=(
            phrase(
                r"\b(?:religie|religion|geloof|faith|moslim|muslim|christen|christian|joods|jewish)\b",
                0.78,
            ),
        ),
    )


def build_sensitive_information_detectors() -> tuple[TextDetector, ...]:
    """Construct sensitive-information detectors in stable order."""

    return (
        build_health_information_detector(),
        build_financial_information_detector(),
        build_political_information_detector(),
        build_religious_information_detector(),
        build_criminal_information_detector(),
        build_minor_information_detector(),
    )
