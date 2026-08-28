"""Train and persist the tokenizer bound to a training snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config.multimodal.training_settings import TrainingSettings


def train_snapshot_tokenizer(
    *,
    training_directory: Path,
    train_samples: tuple[Any, ...],
    snapshot_id: str,
    training_settings: TrainingSettings,
) -> dict[str, object]:
    """Create the versioned tokenizer artifact and checksum sidecar."""

    from multimodal.tokenization.training import (
        TOKENIZER_ARTIFACT_VERSION,
        train_vocabulary_tokenizer,
    )

    records = tuple(
        {
            "record_id": str(sample.sample_id),
            "split": str(sample.split),
            "text": str(sample.text),
        }
        for sample in train_samples
        if sample.text.strip()
    )
    if not records:
        raise ValueError("training snapshot has no train text for tokenizer")

    artifact_path = training_directory / "tokenizer" / "tokenizer.json"
    train_vocabulary_tokenizer(
        records=records,
        output_path=artifact_path,
        vocab_size=int(training_settings.text_tokenizer_vocab_size),
        snapshot_id=snapshot_id,
        seed=int(training_settings.seed),
    )

    artifact_bytes = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    payload = json.loads(artifact_bytes.decode("utf-8"))
    artifact_version = str(payload.get("tokenizer_type") or "")
    if artifact_version != TOKENIZER_ARTIFACT_VERSION:
        raise ValueError("unexpected tokenizer artifact version")

    artifact_path.with_name(artifact_path.name + ".sha256").write_text(
        f"{artifact_sha256}  {artifact_path.name}\n",
        encoding="ascii",
    )
    return {
        "path": artifact_path.relative_to(training_directory).as_posix(),
        "sha256": artifact_sha256,
        "artifact_version": artifact_version,
        "vocab_size": int(payload["vocab_size"]),
        "special_tokens": {
            token: int(payload["token_to_id"][token])
            for token in payload["special_tokens"]
        },
        "corpus": payload["corpus"],
        "trainer": payload["trainer"],
    }


__all__ = ["train_snapshot_tokenizer"]
