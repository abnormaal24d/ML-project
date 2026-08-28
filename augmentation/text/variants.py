"""Text-field augmentation variants as pure functions."""

from __future__ import annotations

from augmentation.text.text_identity import (
    single_line_value,
    text_field_value,
    text_identity,
)


def build_title_variant(
    *,
    text: str,
    title: str | None,
) -> str | None:
    prepared_text = text_field_value(text)
    prepared_title = single_line_value(title)
    if not prepared_text or not prepared_title:
        return None
    if _starts_with_title_boundary(
        text=prepared_text,
        title=prepared_title,
    ):
        return None
    return f"{prepared_title}\n\n{prepared_text}"


def build_context_variant(
    *,
    text: str,
    modality: str | None,
    task_type: str | None,
    domain: str | None,
) -> str | None:
    prepared_text = text_field_value(text)
    if not prepared_text:
        return None
    context = _format_context(
        modality=modality,
        task_type=task_type,
        domain=domain,
    )
    if context is None:
        return None
    context_identity = context.casefold()
    if prepared_text.lstrip().casefold().startswith(context_identity):
        return None
    return f"{context}\n\n{prepared_text}"


def build_span_focus_variant(
    *,
    text: str,
    text_spans: tuple[str, ...],
) -> str | None:
    prepared_text = text_field_value(text)
    if not prepared_text:
        return None
    span = _select_best_span(
        text_spans=text_spans,
        source_text=prepared_text,
    )
    if span is None:
        return None
    return _format_focused_text(span=span, source_text=prepared_text)


def _format_context(
    *,
    modality: str | None,
    task_type: str | None,
    domain: str | None,
) -> str | None:
    lines = _context_lines(
        modality=modality,
        task_type=task_type,
        domain=domain,
    )
    if not lines:
        return None
    return "\n".join(lines)


def _context_lines(
    *,
    modality: str | None,
    task_type: str | None,
    domain: str | None,
) -> tuple[str, ...]:
    lines: list[str] = []
    prepared_task_type = _human_label(task_type)
    prepared_modality = _human_label(modality)
    prepared_domain = single_line_value(domain)
    if prepared_task_type and prepared_modality:
        lines.append(
            f"This {prepared_modality} sample is intended for "
            f"{prepared_task_type}."
        )
    elif prepared_task_type:
        lines.append(f"This sample is intended for {prepared_task_type}.")
    elif prepared_modality:
        lines.append(f"This is a {prepared_modality} sample.")
    if prepared_domain:
        lines.append(f"It comes from {prepared_domain}.")
    return tuple(lines)


def _human_label(value: object) -> str:
    text = single_line_value(value)
    if not text:
        return ""
    return text.replace("_", " ").replace("-", " ").lower()


def _starts_with_title_boundary(*, text: str, title: str) -> bool:
    text_start = text.lstrip()
    if not text_start:
        return False
    folded_text = text_start.casefold()
    folded_title = title.casefold()
    if _has_title_boundary(text=folded_text, title=folded_title):
        return True
    labeled_prefix = "title"
    if not folded_text.startswith(labeled_prefix):
        return False
    remainder = folded_text[len(labeled_prefix) :].lstrip()
    if not remainder.startswith(":"):
        return False
    return _has_title_boundary(
        text=remainder[1:].lstrip(),
        title=folded_title,
    )


def _has_title_boundary(*, text: str, title: str) -> bool:
    if not text.startswith(title):
        return False
    if len(text) == len(title):
        return True
    return text[len(title)].isspace() or text[len(title)] in ":-–—|"


def _select_best_span(
    *,
    text_spans: tuple[str, ...],
    source_text: str,
) -> str | None:
    source_identity = text_identity(source_text)
    seen_identities: set[str] = set()
    best_candidate: tuple[int, int, int, str] | None = None
    for index, raw_span in enumerate(text_spans):
        span = single_line_value(raw_span)
        if not span:
            continue
        span_identity = text_identity(span)
        if not span_identity or span_identity in seen_identities:
            continue
        seen_identities.add(span_identity)
        if span_identity == source_identity:
            continue
        candidate = (_span_score(span), len(span), -index, span)
        if best_candidate is None or candidate > best_candidate:
            best_candidate = candidate
    if best_candidate is None:
        return None
    return best_candidate[3]


def _format_focused_text(*, span: str, source_text: str) -> str | None:
    prepared_span = single_line_value(span)
    prepared_source = text_field_value(source_text)
    if not prepared_span or not prepared_source:
        return None
    if text_identity(prepared_span) == text_identity(prepared_source):
        return None
    return f"Key excerpt:\n{prepared_span}\n\nSource text:\n{prepared_source}"


def _span_score(span: str) -> int:
    token_count = len(span.split())
    char_count = len(span)
    confidence_hint = _confidence_hint(span=span)
    punctuation_bonus = _punctuation_bonus(span=span)
    return token_count * 100 + char_count + confidence_hint + punctuation_bonus


def _confidence_hint(*, span: str) -> int:
    letters = 0
    digits = 0
    for char in span:
        if char.isalpha():
            letters += 1
        elif char.isdigit():
            digits += 1
    if letters == 0:
        return 0
    return min(25, max(0, letters - digits))


def _punctuation_bonus(*, span: str) -> int:
    bonus = 0
    if span.endswith((".", "!", "?")):
        bonus += 10
    if "," in span or ";" in span or ":" in span:
        bonus += 5
    return bonus
