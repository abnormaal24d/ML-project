"""Quality metadata for generated augmentation variants."""

from __future__ import annotations

from augmentation.text.text_identity import text_identity


def evaluate_text_variant_quality(
    *,
    source_text: str,
    variant_text: str,
) -> dict[str, object]:
    source_token_count, source_tokens = _token_stats(source_text)
    variant_token_count, variant_tokens = _token_stats(variant_text)
    jaccard = _jaccard(source_tokens, variant_tokens)
    source_identity = text_identity(source_text)
    variant_identity = text_identity(variant_text)
    source_length = len(source_text)
    variant_length = len(variant_text)
    return {
        "source_char_count": source_length,
        "variant_char_count": variant_length,
        "char_delta": variant_length - source_length,
        "source_token_count": source_token_count,
        "variant_token_count": variant_token_count,
        "token_jaccard_similarity": round(jaccard, 6),
        "duplicate_text": source_identity == variant_identity,
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) <= len(right):
        smaller = left
        larger = right
    else:
        smaller = right
        larger = left
    intersection = sum(1 for token in smaller if token in larger)
    union_size = len(left) + len(right) - intersection
    if union_size <= 0:
        return 0.0
    return intersection / union_size


def _token_stats(text: str) -> tuple[int, set[str]]:
    tokens: set[str] = set()
    token_count = 0
    for token in text.split():
        if not token:
            continue
        token_count += 1
        tokens.add(token.casefold())
    return token_count, tokens
