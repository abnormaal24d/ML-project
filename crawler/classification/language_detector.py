"""Language detection from markup metadata and visible text."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

from logger.project_logger import ProjectLogger
from preprocessing.text.text_language import (
    LanguageDetectionResult as PreprocessingLanguageDetectionResult,
)
from preprocessing.text.text_language import (
    LanguageDetector as HeuristicLanguageDetector,
)

if TYPE_CHECKING:
    from config.settings.classification import (
        LanguageDetectorSettings,
    )


fasttext: ModuleType | None

try:
    fasttext = import_module("fasttext")
except ImportError:  # pragma: no cover
    fasttext = None


_NON_VISIBLE_BLOCK_RE = re.compile(
    r"<(script|style|noscript|template)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

_TAG_RE = re.compile(r"<[^>]+>")

_WHITESPACE_RE = re.compile(r"\s+")

LANGUAGE_DETECTOR_VERSION = "crawler-language-v1"


@dataclass(frozen=True, slots=True)
class LanguageDetection:
    """Resolved language with confidence and provenance."""

    value: str | None
    confidence: float | None = None
    source: str | None = None


class _FastTextModel(Protocol):
    """Subset of a FastText model used by language detection."""

    def predict(
        self,
        text: str,
        *,
        k: int,
    ) -> tuple[
        Sequence[str],
        Sequence[float],
    ]: ...


class _FastTextLoader(Protocol):
    """Callable contract for FastText model loading."""

    def __call__(
        self,
        path: str,
    ) -> _FastTextModel: ...


class LanguageDetector:
    """Resolve language from markup declarations, model evidence and heuristics."""

    def __init__(
        self,
        settings: LanguageDetectorSettings,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._logger = logger

        self._language_re = re.compile(
            settings.language_attribute_pattern,
            re.IGNORECASE,
        )

        self._meta_language_res = tuple(
            re.compile(
                pattern,
                re.IGNORECASE,
            )
            for pattern in settings.meta_language_patterns
        )

        self._fasttext_model_cache: _FastTextModel | None = None

        self._fasttext_load_attempted = False

    def detect(
        self,
        *,
        body: bytes,
        encoding: str | None,
    ) -> LanguageDetection:
        if not body or not encoding:
            return _no_evidence()

        sample = body[: self._settings.sample_size_bytes]

        try:
            markup_text = sample.decode(
                encoding,
                errors="replace",
            )
        except LookupError:
            markup_text = sample.decode(
                "utf-8",
                errors="replace",
            )

        return self.detect_text(markup_text=markup_text)

    def detect_text(
        self,
        *,
        markup_text: str,
    ) -> LanguageDetection:
        if not self._settings.enabled or not markup_text:
            return _no_evidence()

        match = self._language_re.search(markup_text)

        if match:
            return _mapped(
                match.group(1),
                1.0,
                "html_lang",
                markup_text,
            )

        meta = self._detect_meta_language(text=markup_text)

        if meta.value is not None:
            return meta

        visible_text = _visible_text(markup_text)

        fasttext_result = self._detect_fasttext(text=visible_text)

        if fasttext_result.value is not None:
            return fasttext_result

        heuristic = self._detect_heuristic(text=visible_text)

        if heuristic.value is not None:
            return heuristic

        self._logger.debug("language_detector_no_evidence")

        return _no_evidence()

    @property
    def version(self) -> str:
        """Return the stable crawler language detector contract version."""

        return LANGUAGE_DETECTOR_VERSION

    @property
    def _fasttext_model(
        self,
    ) -> _FastTextModel | None:
        if not self._settings.use_fasttext:
            return None

        if not self._fasttext_load_attempted:
            self._fasttext_load_attempted = True
            self._fasttext_model_cache = self._load_fasttext_model()

        return self._fasttext_model_cache

    def _detect_fasttext(
        self,
        *,
        text: str,
    ) -> LanguageDetection:
        model = self._fasttext_model
        stripped = text.strip()

        if model is None or len(stripped) < 32:
            return _no_evidence()

        try:
            labels, scores = model.predict(
                stripped,
                k=1,
            )
        except (
            AttributeError,
            RuntimeError,
            ValueError,
        ):  # pragma: no cover
            return _no_evidence()

        if not labels:
            return _no_evidence()

        confidence = float(scores[0]) if scores else None

        if (
            confidence is not None
            and confidence < self._settings.minimum_confidence
        ):
            return _no_evidence()

        label = labels[0].removeprefix("__label__")

        return _mapped(
            label,
            confidence or 0.0,
            "fasttext",
            stripped,
        )

    def _detect_meta_language(
        self,
        *,
        text: str,
    ) -> LanguageDetection:
        for pattern in self._meta_language_res:
            match = pattern.search(text)

            if match:
                return _mapped(
                    match.group(1),
                    0.95,
                    "html_meta",
                    text,
                )

        return _no_evidence()

    def _detect_heuristic(
        self,
        *,
        text: str,
    ) -> LanguageDetection:
        settings = self._settings.heuristic

        if not settings.enabled:
            return _no_evidence()

        detection = HeuristicLanguageDetector().detect(
            text=text,
            stopwords_by_language=(settings.stopwords_by_language),
            min_text_length=(settings.min_text_length),
            min_best_score=float(settings.min_best_score),
            require_clear_winner=(settings.require_clear_winner),
            use_stopword_matching=(settings.use_stopword_matching),
            use_unicode_ranges=(settings.use_unicode_ranges),
        )

        if detection.language_code in {
            None,
            "und",
        }:
            return _no_evidence()

        return map_language_detection_to_classification_result(detection)

    def _load_fasttext_model(
        self,
    ) -> _FastTextModel | None:
        if fasttext is None:
            self._logger.warning("language_detector_fasttext_unavailable")
            return None

        load_model = vars(fasttext).get("load_model")

        if not callable(load_model):
            self._logger.warning("language_detector_fasttext_unavailable")
            return None

        model_path = self._settings.fasttext_model_path

        if not model_path:
            self._logger.warning("language_detector_fasttext_model_missing")
            return None

        path = Path(model_path).expanduser()

        if not path.exists():
            self._logger.warning(
                "language_detector_fasttext_model_not_found",
                model_path=str(path),
            )
            return None

        try:
            return cast(
                _FastTextLoader,
                load_model,
            )(path.as_posix())
        except (
            AttributeError,
            OSError,
            RuntimeError,
            ValueError,
        ):  # pragma: no cover
            self._logger.warning(
                "language_detector_fasttext_model_load_failed",
                model_path=str(path),
            )
            return None


def map_language_detection_to_classification_result(
    detection: PreprocessingLanguageDetectionResult,
) -> LanguageDetection:
    """Map preprocessing language evidence to crawler language evidence."""

    if detection.language_code is None:
        return _no_evidence()

    return LanguageDetection(
        value=detection.language_code,
        confidence=detection.confidence,
        source=detection.detector_name,
    )


def _mapped(
    value: str,
    confidence: float,
    source: str,
    text: str,
) -> LanguageDetection:
    return map_language_detection_to_classification_result(
        PreprocessingLanguageDetectionResult(
            language_code=(_normalize_language(value)),
            confidence=confidence,
            detector_name=source,
            text_length=len(text.strip()),
        )
    )


def _visible_text(
    markup_text: str,
) -> str:
    without_blocks = _NON_VISIBLE_BLOCK_RE.sub(
        " ",
        markup_text,
    )

    without_tags = _TAG_RE.sub(
        " ",
        without_blocks,
    )

    return _WHITESPACE_RE.sub(
        " ",
        html.unescape(without_tags),
    ).strip()


def _normalize_language(
    value: str,
) -> str:
    text = value.strip().casefold()

    if not text:
        return ""

    return (
        re.split(
            r"[_\-,\s]+",
            text,
            maxsplit=1,
        )[0]
        or text
    )


def _no_evidence() -> LanguageDetection:
    return LanguageDetection(
        None,
        None,
        None,
    )
