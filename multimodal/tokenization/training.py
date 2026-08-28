"""Deterministic byte-level BPE training for the canonical tokenizer."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast

from multimodal.tokenization.merges import apply_merge, symbol_key

SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<mask>",
    "<image>",
    "<audio>",
    "<video>",
    "<doc>",
    "<user>",
    "<assistant>",
    "<system>",
    "<tool>",
)
TOKENIZER_ARTIFACT_VERSION = "byte_bpe_v2"
MINIMUM_BYTE_BPE_VOCAB_SIZE = len(SPECIAL_TOKENS) + 256
NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


def train_vocabulary_tokenizer(
    *,
    output_path: Path,
    vocab_size: int,
    records: Sequence[Mapping[str, str]] | None = None,
    texts: Sequence[str] | None = None,
    snapshot_id: str | None = None,
    seed: int = 42,
    normalization: str = "NFC",
) -> None:
    """Learn deterministic byte-level BPE merges from train records only.

    Each record mapping must provide ``record_id``, ``split``, and ``text``.
    """

    training_records = _training_records(
        records=records,
        texts=texts,
        snapshot_id=snapshot_id,
    )
    normalization = _validate_training_schema(
        records=training_records,
        vocab_size=vocab_size,
        normalization=normalization,
    )
    normalized_texts = [
        unicodedata.normalize(normalization, record["text"])
        for record in training_records
    ]
    sequences = [
        [bytes((value,)) for value in text.encode("utf-8")]
        for text in normalized_texts
        if text
    ]
    if not sequences:
        raise ValueError("tokenizer training corpus must contain text")

    merges, learned_symbols = _learn_merges(
        sequences=sequences,
        maximum_symbols=vocab_size - len(SPECIAL_TOKENS) - 256,
    )
    token_to_id = _token_mapping(
        vocab_size=vocab_size,
        learned_symbols=learned_symbols,
    )
    payload = _artifact_payload(
        records=training_records,
        normalized_texts=normalized_texts,
        snapshot_id=snapshot_id,
        seed=seed,
        normalization=normalization,
        token_to_id=token_to_id,
        merges=merges,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _training_records(
    *,
    records: Sequence[Mapping[str, str]] | None,
    texts: Sequence[str] | None,
    snapshot_id: str | None,
) -> tuple[dict[str, str], ...]:
    if records is not None and texts is not None:
        raise ValueError("pass records or texts, not both")
    if records is not None:
        return tuple(dict(record) for record in records)
    if texts is None:
        raise ValueError("tokenizer training records are required")
    prefix = snapshot_id or "direct-train-corpus"
    return tuple(
        {
            "record_id": f"{prefix}:{index}",
            "split": "train",
            "text": str(text),
        }
        for index, text in enumerate(texts)
    )


def _validate_training_schema(
    *,
    records: tuple[Mapping[str, str], ...],
    vocab_size: int,
    normalization: str,
) -> NormalizationForm:
    if vocab_size < MINIMUM_BYTE_BPE_VOCAB_SIZE:
        raise ValueError(
            "byte-level BPE vocab_size must be at least "
            f"{MINIMUM_BYTE_BPE_VOCAB_SIZE}"
        )
    if normalization not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError("unsupported Unicode normalization")
    if not records:
        raise ValueError("tokenizer training records must not be empty")
    record_ids = [
        str(record.get("record_id", "")).strip() for record in records
    ]
    if any(not record_id for record_id in record_ids):
        raise ValueError("tokenizer training record_id must not be blank")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("tokenizer training record_ids must be unique")
    foreign_splits = sorted(
        {
            str(record.get("split", "")).strip().lower()
            for record in records
            if str(record.get("split", "")).strip().lower() != "train"
        }
    )
    if foreign_splits:
        raise ValueError(
            "tokenizer training accepts only the train split; found "
            f"{foreign_splits}"
        )
    return cast(NormalizationForm, normalization)


def _learn_merges(
    *,
    sequences: list[list[bytes]],
    maximum_symbols: int,
) -> tuple[list[tuple[bytes, bytes, bytes]], list[bytes]]:
    merges: list[tuple[bytes, bytes, bytes]] = []
    learned_symbols: list[bytes] = []
    known = {bytes((value,)) for value in range(256)}
    while len(learned_symbols) < maximum_symbols:
        counts: Counter[tuple[bytes, bytes]] = Counter()
        for sequence in sequences:
            counts.update(zip(sequence, sequence[1:], strict=False))
        if not counts:
            break
        left, right = min(
            counts,
            key=lambda pair: (-counts[pair], pair[0], pair[1]),
        )
        merged = left + right
        sequences = [
            apply_merge(sequence, left=left, right=right, merged=merged)
            for sequence in sequences
        ]
        merges.append((left, right, merged))
        if merged not in known:
            known.add(merged)
            learned_symbols.append(merged)
    return merges, learned_symbols


def _token_mapping(
    *,
    vocab_size: int,
    learned_symbols: list[bytes],
) -> dict[str, int]:
    mapping = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for symbol in (
        *(bytes((value,)) for value in range(256)),
        *learned_symbols,
    ):
        key = symbol_key(symbol)
        if key not in mapping and len(mapping) < vocab_size:
            mapping[key] = len(mapping)
    unused_index = 0
    while len(mapping) < vocab_size:
        mapping[f"<unused_{unused_index:06d}>"] = len(mapping)
        unused_index += 1
    return mapping


def _artifact_payload(
    *,
    records: tuple[Mapping[str, str], ...],
    normalized_texts: list[str],
    snapshot_id: str | None,
    seed: int,
    normalization: str,
    token_to_id: dict[str, int],
    merges: list[tuple[bytes, bytes, bytes]],
) -> dict[str, object]:
    record_ids = sorted(str(record["record_id"]) for record in records)
    record_ids_hash = hashlib.sha256(
        "\n".join(record_ids).encode("utf-8")
    ).hexdigest()
    return {
        "tokenizer_type": TOKENIZER_ARTIFACT_VERSION,
        "algorithm": "byte_level_bpe",
        "normalization": normalization,
        "preserve_case": True,
        "vocab_size": len(token_to_id),
        "special_tokens": list(SPECIAL_TOKENS),
        "token_to_id": token_to_id,
        "byte_tokens": [symbol_key(bytes((value,))) for value in range(256)],
        "merges": [
            [symbol_key(left), symbol_key(right), symbol_key(merged)]
            for left, right, merged in merges
        ],
        "trainer": {"seed": seed, "requested_vocab_size": len(token_to_id)},
        "corpus": {
            "snapshot_id": snapshot_id or "unbound-training-corpus",
            "split": "train",
            "record_ids": record_ids,
            "record_ids_sha256": record_ids_hash,
            "record_count": len(records),
            "unicode_characters": sum(len(text) for text in normalized_texts),
            "utf8_bytes": sum(
                len(text.encode("utf-8")) for text in normalized_texts
            ),
        },
    }
