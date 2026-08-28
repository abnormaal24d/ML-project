"""Consolidated credentials privacy detectors."""

from preprocessing.privacy.inspection.detector import TextDetector
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.pattern_detector import (
    PatternDetector,
    PatternSpec,
    compile_pattern,
)


def build_api_credential_detector() -> PatternDetector:
    return PatternDetector(
        name="api_credential",
        version="1.0.0",
        specifications=(
            PatternSpec(
                FindingType.API_CREDENTIAL,
                compile_pattern(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
                0.995,
            ),
            PatternSpec(
                FindingType.API_CREDENTIAL,
                compile_pattern(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
                0.995,
            ),
            PatternSpec(
                FindingType.API_CREDENTIAL,
                compile_pattern(
                    r"(?:api[_ -]?key|client[_ -]?secret|secret[_ -]?key)"
                    r'\s*[:=]\s*[\'"]?[A-Za-z0-9_./+~-]{16,}',
                    ignore_case=True,
                ),
                0.92,
            ),
            PatternSpec(
                FindingType.API_CREDENTIAL,
                compile_pattern(
                    r"(?:password|passwd|secret|token)"
                    r'\s*[:=]\s*[\'"]?[A-Za-z0-9_./+~-]{12,}',
                    ignore_case=True,
                ),
                0.90,
            ),
        ),
    )


def build_basic_auth_detector() -> PatternDetector:
    return PatternDetector(
        name="basic_auth_credential",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.BASIC_AUTH_CREDENTIAL,
                pattern=compile_pattern(
                    r"\bBasic\s+[A-Za-z0-9+/]{8,}={0,2}",
                    ignore_case=True,
                ),
                confidence=0.98,
            ),
        ),
    )


def build_cloud_credential_detector() -> PatternDetector:
    return PatternDetector(
        name="cloud_credential",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.CLOUD_CREDENTIAL,
                pattern=compile_pattern(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
                confidence=0.999,
            ),
            PatternSpec(
                finding_type=FindingType.CLOUD_CREDENTIAL,
                pattern=compile_pattern(r"\bAIza[0-9A-Za-z_-]{35}\b"),
                confidence=0.999,
            ),
        ),
    )


def build_database_credential_detector() -> PatternDetector:
    return PatternDetector(
        name="database_credential",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.DATABASE_CREDENTIAL,
                pattern=compile_pattern(
                    r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://"
                    r"[^\s:/]+:[^\s@]+@[^\s]+",
                    ignore_case=True,
                ),
                confidence=0.99,
            ),
        ),
    )


def build_jwt_detector() -> PatternDetector:
    return PatternDetector(
        name="jwt_token",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.JWT_TOKEN,
                pattern=compile_pattern(
                    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}\."
                    r"[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"
                    r"(?![A-Za-z0-9_-])"
                ),
                confidence=0.99,
            ),
        ),
    )


def build_oauth_token_detector() -> PatternDetector:
    return PatternDetector(
        name="oauth_token",
        version="1.0.0",
        specifications=(
            PatternSpec(
                FindingType.OAUTH_TOKEN,
                compile_pattern(
                    r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}",
                    ignore_case=True,
                ),
                0.98,
            ),
            PatternSpec(
                FindingType.OAUTH_TOKEN,
                compile_pattern(
                    r'(?:access|refresh)[_-]?token\s*[:=]\s*[\'"]?'
                    r"[A-Za-z0-9._~+/=-]{20,}",
                    ignore_case=True,
                ),
                0.94,
            ),
        ),
    )


def build_private_key_detector() -> PatternDetector:
    return PatternDetector(
        name="private_key",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.PRIVATE_KEY,
                pattern=compile_pattern(
                    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
                ),
                confidence=1.0,
            ),
        ),
    )


def build_session_credential_detector() -> PatternDetector:
    return PatternDetector(
        name="session_credential",
        version="1.0.0",
        specifications=(
            PatternSpec(
                FindingType.SESSION_CREDENTIAL,
                compile_pattern(
                    r"(?:session(?:id|_id)?|cookie|csrf(?:_token)?)"
                    r'\s*[:=]\s*[\'"]?[A-Za-z0-9._~+/=-]{12,}',
                    ignore_case=True,
                ),
                0.92,
            ),
        ),
    )


def build_credential_detectors() -> tuple[TextDetector, ...]:
    """Construct the default credential detector family in stable order."""

    return (
        build_api_credential_detector(),
        build_cloud_credential_detector(),
        build_oauth_token_detector(),
        build_jwt_detector(),
        build_session_credential_detector(),
        build_private_key_detector(),
        build_basic_auth_detector(),
        build_database_credential_detector(),
    )
