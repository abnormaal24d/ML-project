"""Persistent, decodeable byte-level BPE tokenizer."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from multimodal.tokenization.merges import apply_merge, symbol_key

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


@dataclass(frozen=True, slots=True)
class VocabularyTokenizer:
    """Canonical runtime tokenizer loaded from a trained BPE artifact."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    max_tokens: int
    token_bytes: dict[int, bytes]
    merges: tuple[tuple[bytes, bytes, bytes], ...]
    normalization: NormalizationForm = "NFC"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @classmethod
    def load(cls, path: Path, *, max_tokens: int) -> "VocabularyTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("algorithm") != "byte_level_bpe":
            raise ValueError("tokenizer artifact must use byte_level_bpe")
        token_to_id = {
            str(token): int(idx)
            for token, idx in payload["token_to_id"].items()
        }
        token_bytes = {
            token_to_id[token]: _parse_symbol(token)
            for token in token_to_id
            if token.startswith("<bytes:")
        }
        merges = tuple(
            (
                _parse_symbol(str(row[0])),
                _parse_symbol(str(row[1])),
                _parse_symbol(str(row[2])),
            )
            for row in payload.get("merges", [])
        )
        tokenizer = cls(
            token_to_id=token_to_id,
            id_to_token={idx: token for token, idx in token_to_id.items()},
            max_tokens=max_tokens,
            token_bytes=token_bytes,
            merges=merges,
            normalization=_read_normalization(payload.get("normalization")),
        )
        tokenizer._validate_byte_coverage()
        return tokenizer

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        pad_to_max_length: bool = True,
    ) -> list[int]:
        normalized = unicodedata.normalize(self.normalization, text)
        symbols = [bytes((value,)) for value in normalized.encode("utf-8")]
        for left, right, merged in self.merges:
            symbols = apply_merge(
                symbols,
                left=left,
                right=right,
                merged=merged,
            )
        ids = [self.token_to_id[symbol_key(symbol)] for symbol in symbols]
        if add_special_tokens:
            ids = [self.token_to_id[self.bos_token], *ids]
            if len(ids) < self.max_tokens:
                ids.append(self.token_to_id[self.eos_token])
        ids = ids[: self.max_tokens]
        if pad_to_max_length:
            pad_id = self.token_to_id[self.pad_token]
            ids.extend([pad_id] * (self.max_tokens - len(ids)))
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        skipped = {
            self.token_to_id[self.pad_token],
            self.token_to_id[self.bos_token],
            self.token_to_id[self.eos_token],
        }
        payload = b"".join(
            self.token_bytes[int(token_id)]
            for token_id in ids
            if int(token_id) not in skipped
            and int(token_id) in self.token_bytes
        )
        return payload.decode("utf-8", errors="replace")

    def _validate_byte_coverage(self) -> None:
        missing = [
            value
            for value in range(256)
            if symbol_key(bytes((value,))) not in self.token_to_id
        ]
        if missing:
            raise ValueError(
                "byte-level tokenizer artifact has incomplete byte coverage"
            )


def _parse_symbol(token: str) -> bytes:
    if not token.startswith("<bytes:") or not token.endswith(">"):
        raise ValueError(f"invalid byte token: {token!r}")
    try:
        return bytes.fromhex(token[7:-1])
    except ValueError as exc:
        raise ValueError(f"invalid byte token: {token!r}") from exc


def _read_normalization(value: object) -> NormalizationForm:
    normalization = str(value or "NFC")
    if normalization not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError("tokenizer artifact has unsupported normalization")
    return cast(NormalizationForm, normalization)
