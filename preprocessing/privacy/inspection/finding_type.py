"""Canonical categories emitted by privacy detectors."""

from enum import StrEnum


class FindingType(StrEnum):
    PERSON_NAME = "person_name"
    EMAIL_ADDRESS = "email_address"
    PHONE_NUMBER = "phone_number"
    POSTAL_ADDRESS = "postal_address"
    GEOGRAPHIC_LOCATION = "geographic_location"
    DATE_OF_BIRTH = "date_of_birth"
    BELGIAN_NATIONAL_NUMBER = "belgian_national_number"
    PASSPORT_NUMBER = "passport_number"
    IDENTITY_DOCUMENT = "identity_document"
    IBAN = "iban"
    PAYMENT_CARD = "payment_card"
    IP_ADDRESS = "ip_address"
    LICENSE_PLATE = "license_plate"
    FACE = "face"
    SIGNATURE = "signature"
    MACHINE_READABLE_CODE = "machine_readable_code"
    ORGANIZATION = "organization"
    HEALTH_INFORMATION = "health_information"
    FINANCIAL_INFORMATION = "financial_information"
    POLITICAL_INFORMATION = "political_information"
    RELIGIOUS_INFORMATION = "religious_information"
    CRIMINAL_INFORMATION = "criminal_information"
    MINOR_INFORMATION = "minor_information"
    API_CREDENTIAL = "api_credential"
    CLOUD_CREDENTIAL = "cloud_credential"
    OAUTH_TOKEN = "oauth_token"  # nosec: B105
    JWT_TOKEN = "jwt_token"  # nosec: B105
    SESSION_CREDENTIAL = "session_credential"
    PRIVATE_KEY = "private_key"
    BASIC_AUTH_CREDENTIAL = "basic_auth_credential"
    DATABASE_CREDENTIAL = "database_credential"
    VOICE_IDENTITY = "voice_identity"
