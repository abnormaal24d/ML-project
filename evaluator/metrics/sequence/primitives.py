"""Pure sequence metric primitives."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multimodal.tokenization.text import VocabularyTokenizer


def _normalize_evaluation_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _clean_token_ids(
    values: Any,
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    ignore_label: int | None,
) -> list[int]:
    cleaned: list[int] = []
    for raw_value in values.detach().cpu().tolist():
        value = int(raw_value)
        if ignore_label is not None and value == ignore_label:
            continue
        if value == eos_id:
            break
        if value in {pad_id, bos_id}:
            continue
        cleaned.append(value)
    return cleaned


def _decode_evaluation_text(
    *,
    token_ids: list[int],
    tokenizer: VocabularyTokenizer | None = None,
) -> str:
    if tokenizer is not None:
        return tokenizer.decode(token_ids)
    return " ".join(str(token_id) for token_id in token_ids)


def _token_f1(*, prediction: list[int], target: list[int]) -> float:
    if not prediction and not target:
        return 1.0
    if not prediction or not target:
        return 0.0
    overlap = sum((Counter(prediction) & Counter(target)).values())
    precision = overlap / len(prediction)
    recall = overlap / len(target)
    return (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )


def _rouge_l_f1(*, prediction: list[str], target: list[str]) -> float:
    if not prediction and not target:
        return 1.0
    if not prediction or not target:
        return 0.0
    overlap = _longest_common_subsequence(prediction, target)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(target)
    return 2.0 * precision * recall / (precision + recall)


def _longest_common_subsequence(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _edit_distance(*, prediction: list[str], target: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for target_index, target_value in enumerate(target, start=1):
        current = [target_index]
        for prediction_index, prediction_value in enumerate(
            prediction, start=1
        ):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1]
                    + int(target_value != prediction_value),
                )
            )
        previous = current
    return previous[-1]


IGNORE_LABEL = -100


def _special_token_ids(
    tokenizer: VocabularyTokenizer | None = None,
) -> tuple[int, int, int]:
    if tokenizer is None:
        return 0, 2, 3
    return (
        int(tokenizer.token_to_id[tokenizer.pad_token]),
        int(tokenizer.token_to_id[tokenizer.bos_token]),
        int(tokenizer.token_to_id[tokenizer.eos_token]),
    )


__all__ = [
    "_clean_token_ids",
    "_decode_evaluation_text",
    "_normalize_evaluation_text",
    "_token_f1",
    "_rouge_l_f1",
    "_longest_common_subsequence",
    "_edit_distance",
    "_special_token_ids",
    "IGNORE_LABEL",
]
