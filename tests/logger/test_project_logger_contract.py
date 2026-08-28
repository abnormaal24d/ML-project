"""Structured record contract for the ProjectLogger facade."""

from __future__ import annotations

import logging
import uuid

import pytest

from logger.project_logger import ProjectLogger


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured() -> tuple[ProjectLogger, list[logging.LogRecord]]:
    logger = logging.getLogger(f"test.project-logger.{uuid.uuid4().hex}")
    logger.setLevel(logging.DEBUG)
    handler = _RecordHandler()
    logger.addHandler(handler)
    yield ProjectLogger(logger), handler.records
    logger.removeHandler(handler)


def test_event_name_is_first_positional_argument(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.info("task_started")
    assert records[0]._project_event == "task_started"


def test_event_name_is_redacted(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.info("fetched https://example.test/a?token=secret")
    assert "example.test" in records[0]._project_event
    assert "secret" not in records[0]._project_event


def test_message_kwarg_becomes_project_message(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.info("event_name", message="readable text")
    assert records[0]._project_message == "readable text"


def test_message_kwarg_is_redacted(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.info(
        "event_name",
        message=r"stored at C:\Users\abnor\data\raw.json",
    )
    assert records[0]._project_message == "stored at [LOCAL_PATH]"


def test_structured_fields_become_record_attributes(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.info("event_name", task_id="task-1", stage="persistence")
    record = records[0]
    assert record.task_id == "task-1"
    assert record.stage == "persistence"


def test_field_values_are_redacted(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.info("event_name", api_key="sk-123", password="pw")
    assert records[0].api_key == "[REDACTED]"
    assert records[0].password == "[REDACTED]"


def test_nested_mapping_fields_are_redacted(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.info(
        "event_name",
        metadata={"quality_score": 0.8, "token": "abc"},
    )
    assert records[0].metadata == {"quality_score": 0.8, "token": "[REDACTED]"}


def test_extra_must_be_a_mapping(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, _ = captured
    with pytest.raises(TypeError, match="logger extra must be a mapping"):
        project_logger.info("event_name", extra="not-a-mapping")


def test_static_context_fields_are_included(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    _logger, records = captured
    project_logger = ProjectLogger(
        _logger._logger, context={"source_id": "s1"}
    )
    project_logger.info("event_name", task_id="t1")
    record = records[0]
    assert record.source_id == "s1"
    assert record.task_id == "t1"


def test_field_keys_are_tracked(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.info("event_name", a=1, b=2)
    assert records[0]._project_field_keys == ("a", "b")


def test_exception_attaches_exc_info(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger.exception("handler_exception", reason="boom")
    assert records[0].exc_info is not None


def test_disabled_level_emits_nothing(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger._logger.setLevel(logging.INFO)
    project_logger.debug("debug_event", payload=b"x")
    assert records == []


def test_debug_lazy_skips_builders_when_disabled(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    project_logger._logger.setLevel(logging.INFO)
    built: list[str] = []

    def builder() -> object:
        built.append("built")
        return {"expensive": 1}

    project_logger.debug_lazy("debug_event", field_builders={"data": builder})
    assert built == []
    assert records == []


def test_debug_lazy_builds_when_enabled(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, records = captured
    built: list[str] = []

    def builder() -> object:
        built.append("built")
        return {"expensive": 1}

    project_logger.debug_lazy("debug_event", field_builders={"data": builder})
    assert built == ["built"]
    assert records[0].data == {"expensive": 1}


def test_name_property_delegates_to_underlying_logger(
    captured: tuple[ProjectLogger, list[logging.LogRecord]],
) -> None:
    project_logger, _ = captured
    assert project_logger.name == project_logger._logger.name
