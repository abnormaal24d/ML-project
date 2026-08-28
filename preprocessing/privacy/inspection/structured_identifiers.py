"""Consolidated structured-identifiers privacy detectors."""

from __future__ import annotations

import ipaddress
import re
from datetime import date

from preprocessing.privacy.inspection.detector import (
    TextDetector,
    TextDetectorInput,
)
from preprocessing.privacy.inspection.evidence_location import (
    EvidenceLocation,
    TextSpan,
)
from preprocessing.privacy.inspection.finding import (
    PrivacyFinding,
    stable_finding_id,
)
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.pattern_detector import (
    PatternDetector,
    PatternSpec,
    compile_pattern,
)
from preprocessing.privacy.inspection.value_digest import (
    digest_sensitive_value,
)


def normalize_belgian_national_number(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def valid_belgian_national_number(value: str) -> bool:
    normalized = normalize_belgian_national_number(value)
    if len(normalized) != 11:
        return False
    birth, serial, check = normalized[:6], normalized[6:9], int(normalized[9:])
    if serial == "000":
        return False
    month = int(birth[2:4])
    day = int(birth[4:6])
    plausible = False
    for century in (1900, 2000):
        try:
            date(century + int(birth[:2]), month, day)
            plausible = True
        except ValueError:
            pass
    if not plausible:
        return False
    base = int(birth + serial)
    return check in {97 - base % 97, 97 - int("2" + birth + serial) % 97}


def build_belgian_national_number_detector() -> PatternDetector:
    return PatternDetector(
        name="belgian_national_number",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.BELGIAN_NATIONAL_NUMBER,
                pattern=compile_pattern(
                    r"(?<!\d)\d{2}[. /-]?\d{2}[. /-]?\d{2}"
                    r"[. /-]?\d{3}[. /-]?\d{2}(?!\d)"
                ),
                confidence=0.999,
                validator=valid_belgian_national_number,
                normalizer=normalize_belgian_national_number,
                country="BE",
            ),
        ),
    )


_CONTEXT = re.compile(
    r"(?:geboren(?: op)?|geboortedatum|date de naissance|born(?: on)?|"
    r"birth date|geburtsdatum)\s*[:=-]?\s*"
    r"(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
    re.IGNORECASE,
)


class DateOfBirthDetector:
    name = "date_of_birth"
    version = "1.0.0"

    def detect(self, item: TextDetectorInput) -> tuple[PrivacyFinding, ...]:
        findings = []
        for match in _CONTEXT.finditer(item.text):
            span = match.span("date")
            value = match.group("date")
            location = EvidenceLocation(
                field_name=item.field_name,
                text_span=TextSpan(*span),
                page_number=item.page_number,
            )
            value_digest = digest_sensitive_value(value)
            findings.append(
                PrivacyFinding(
                    finding_id=stable_finding_id(
                        finding_type=FindingType.DATE_OF_BIRTH,
                        detector_name=self.name,
                        detector_version=self.version,
                        location=location,
                        normalized_value_digest=value_digest,
                    ),
                    finding_type=FindingType.DATE_OF_BIRTH,
                    confidence=0.94,
                    location=location,
                    detector_name=self.name,
                    detector_version=self.version,
                    normalized_value_digest=value_digest,
                    country=item.country,
                    language=item.resolved_language(),
                )
            )
        return tuple(findings)


def build_date_of_birth_detector() -> DateOfBirthDetector:
    return DateOfBirthDetector()


def build_email_address_detector() -> PatternDetector:
    return PatternDetector(
        name="email_address",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.EMAIL_ADDRESS,
                pattern=compile_pattern(
                    r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
                    r"@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
                    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+",
                    ignore_case=True,
                ),
                confidence=0.99,
            ),
        ),
    )


def normalize_iban(value: str) -> str:
    return "".join(
        character for character in value if character.isalnum()
    ).upper()


def valid_iban(value: str) -> bool:
    iban = normalize_iban(value)
    if not 15 <= len(iban) <= 34 or not iban[:2].isalpha():
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(
        str(ord(character) - 55) if character.isalpha() else character
        for character in rearranged
    )
    remainder = 0
    for character in numeric:
        remainder = (remainder * 10 + int(character)) % 97
    return remainder == 1


def build_iban_detector() -> PatternDetector:
    return PatternDetector(
        name="iban",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.IBAN,
                pattern=compile_pattern(
                    r"(?<![A-Z0-9])BE\d{2}(?:[ -]?\d){12}(?![A-Z0-9])",
                    ignore_case=True,
                ),
                confidence=0.999,
                validator=valid_iban,
                normalizer=normalize_iban,
                country="BE",
            ),
            PatternSpec(
                finding_type=FindingType.IBAN,
                pattern=compile_pattern(
                    r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{11,30}"
                    r"(?![A-Z0-9])",
                    ignore_case=True,
                ),
                confidence=0.995,
                validator=valid_iban,
                normalizer=normalize_iban,
            ),
        ),
    )


_CANDIDATE = re.compile(
    r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])"
    r"|(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"
)


class IpAddressDetector:
    name = "ip_address"
    version = "1.0.0"

    def detect(self, item: TextDetectorInput) -> tuple[PrivacyFinding, ...]:
        findings: list[PrivacyFinding] = []
        for match in _CANDIDATE.finditer(item.text):
            value = match.group(0).strip()
            try:
                address = ipaddress.ip_address(value)
            except ValueError:
                continue
            location = EvidenceLocation(
                field_name=item.field_name,
                text_span=TextSpan(match.start(), match.end()),
                page_number=item.page_number,
            )
            value_digest = digest_sensitive_value(address.compressed)
            findings.append(
                PrivacyFinding(
                    finding_id=stable_finding_id(
                        finding_type=FindingType.IP_ADDRESS,
                        detector_name=self.name,
                        detector_version=self.version,
                        location=location,
                        normalized_value_digest=value_digest,
                    ),
                    finding_type=FindingType.IP_ADDRESS,
                    confidence=0.99,
                    location=location,
                    detector_name=self.name,
                    detector_version=self.version,
                    normalized_value_digest=value_digest,
                    language=item.resolved_language(),
                    attributes={
                        "version": address.version,
                        "private": address.is_private,
                    },
                )
            )
        return tuple(findings)


def build_ip_address_detector() -> IpAddressDetector:
    return IpAddressDetector()


_PATTERN = re.compile(
    r"(?:passport|paspoort|passeport|reisepass)(?:nummer|number|no|nr)?"
    r"\s*[:#=-]?\s*(?P<value>[A-Z0-9]{6,12})",
    re.IGNORECASE,
)


class PassportNumberDetector:
    name = "passport_number"
    version = "1.0.0"

    def detect(self, item: TextDetectorInput) -> tuple[PrivacyFinding, ...]:
        findings = []
        for match in _PATTERN.finditer(item.text):
            start, end = match.span("value")
            value = match.group("value")
            location = EvidenceLocation(
                field_name=item.field_name,
                text_span=TextSpan(start, end),
                page_number=item.page_number,
            )
            value_digest = digest_sensitive_value(value)
            findings.append(
                PrivacyFinding(
                    finding_id=stable_finding_id(
                        finding_type=FindingType.PASSPORT_NUMBER,
                        detector_name=self.name,
                        detector_version=self.version,
                        location=location,
                        normalized_value_digest=value_digest,
                    ),
                    finding_type=FindingType.PASSPORT_NUMBER,
                    confidence=0.93,
                    location=location,
                    detector_name=self.name,
                    detector_version=self.version,
                    normalized_value_digest=value_digest,
                    country=item.country,
                    language=item.resolved_language(),
                )
            )
        return tuple(findings)


def build_passport_number_detector() -> PassportNumberDetector:
    return PassportNumberDetector()


def _digits(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def valid_luhn(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        number = int(character)
        if index % 2 == parity:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def build_payment_card_detector() -> PatternDetector:
    return PatternDetector(
        name="payment_card",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.PAYMENT_CARD,
                pattern=compile_pattern(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
                confidence=0.995,
                validator=valid_luhn,
                normalizer=_digits,
            ),
        ),
    )


_DATE_ONLY = re.compile(r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$")


def _valid_phone(value: str) -> bool:
    if _DATE_ONLY.fullmatch(value.strip()):
        return False
    digits = "".join(character for character in value if character.isdigit())
    if not 8 <= len(digits) <= 15:
        return False
    if len(set(digits)) == 1:
        return False
    return True


def build_phone_number_detector() -> PatternDetector:
    return PatternDetector(
        name="phone_number",
        version="1.0.0",
        specifications=(
            PatternSpec(
                finding_type=FindingType.PHONE_NUMBER,
                pattern=compile_pattern(
                    r"(?<![\w])(?:\+|00)?\d(?:[ .()/-]?\d){7,14}(?![\w])"
                ),
                confidence=0.86,
                validator=_valid_phone,
            ),
        ),
    )


def build_structured_identifier_detectors() -> tuple[TextDetector, ...]:
    """Construct structured-identifier detectors in stable order."""

    return (
        build_email_address_detector(),
        build_iban_detector(),
        build_payment_card_detector(),
        build_belgian_national_number_detector(),
        build_date_of_birth_detector(),
        build_passport_number_detector(),
        build_ip_address_detector(),
        build_phone_number_detector(),
    )
