"""Collate document text, layout boxes, and table structure tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mmcrawler_datasets.collation.tensor_ops import (
    FEATURE_DTYPE,
    LABEL_DTYPE,
    MASK_DTYPE,
    stack_feature_matrix,
    to_float_tensor,
)
from multimodal.tokenization.text import VocabularyTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mmcrawler_datasets.schema import MultimodalSample


def container_length(value: object) -> int:
    """Return a canonical container length and reject malformed values."""

    if value is None:
        return 0
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value)
    raise TypeError(
        f"structured table values must be containers, got {type(value).__name__}"
    )


class DocumentCollator:
    """Build document token tensors and structured layout inputs."""

    def __init__(
        self,
        *,
        tokenizer: VocabularyTokenizer,
    ) -> None:
        """Store document tokenization limits for collation."""
        self._tokenizer = tokenizer

    def collate_sample(self, sample: MultimodalSample) -> torch.Tensor:
        """Tokenize document text for one sample."""
        document_text = sample.text or sample.input_text or ""
        if not sample.has_document:
            document_text = ""
        ids = self._tokenizer.encode(document_text)
        return torch.tensor(ids, dtype=torch.long)

    def collate_batch(
        self, samples: Sequence[MultimodalSample]
    ) -> torch.Tensor:
        """Stack document token tensors for a batch of samples."""
        return stack_feature_matrix(
            [self.collate_sample(sample) for sample in samples]
        )

    @staticmethod
    def collate_box_targets(
        values: Sequence[Sequence[object]],
    ) -> torch.Tensor:
        """Average bounding boxes per sample into fixed-size targets."""
        targets = []
        for sample_annotations in values:
            boxes = []
            for annotation in sample_annotations:
                box = getattr(annotation, "box", None)
                if box is None:
                    continue
                boxes.append(
                    [
                        float(getattr(box, "x", 0.0)),
                        float(getattr(box, "y", 0.0)),
                        float(getattr(box, "width", 0.0)),
                        float(getattr(box, "height", 0.0)),
                    ]
                )
            if boxes:
                targets.append(
                    to_float_tensor(
                        [
                            coordinate
                            for box_coordinates in boxes
                            for coordinate in box_coordinates
                        ]
                    )
                    .reshape(-1, 4)
                    .mean(dim=0)
                )
            else:
                targets.append(torch.zeros(4, dtype=FEATURE_DTYPE))
        return stack_feature_matrix(targets)

    @staticmethod
    def collate_layout_inputs(
        values: Sequence[Sequence[object]],
        *,
        max_boxes: int = 64,
    ) -> dict[str, torch.Tensor]:
        """Pad layout boxes, page ids, and attention masks for a batch."""
        observed = max(
            (len(sample_boxes) for sample_boxes in values), default=0
        )
        box_count = max(1, min(max_boxes, observed))
        boxes = torch.zeros(len(values), box_count, 4, dtype=FEATURE_DTYPE)
        page_ids = torch.zeros(len(values), box_count, dtype=LABEL_DTYPE)
        attention = torch.zeros(len(values), box_count, dtype=MASK_DTYPE)

        for row_index, sample_boxes in enumerate(values):
            write_index = 0
            for annotation in sample_boxes:
                if write_index >= box_count:
                    break
                box = getattr(annotation, "box", None)
                if box is None:
                    continue
                boxes[row_index, write_index] = to_float_tensor(
                    [
                        float(getattr(box, "x", 0.0)),
                        float(getattr(box, "y", 0.0)),
                        float(getattr(box, "width", 0.0)),
                        float(getattr(box, "height", 0.0)),
                    ]
                )
                page = getattr(box, "page", None)
                page_ids[row_index, write_index] = max(0, int(page or 1) - 1)
                attention[row_index, write_index] = True
                write_index += 1

        return {
            "document_layout_boxes": boxes,
            "document_page_ids": page_ids,
            "document_layout_attention_mask": attention,
        }

    @staticmethod
    def collate_table_inputs(
        samples: Sequence[MultimodalSample],
    ) -> torch.Tensor:
        """Encode table structure metadata as per-sample feature rows."""
        rows: list[torch.Tensor] = []
        for sample in samples:
            table = sample.target_table_structure
            if isinstance(table, dict):
                row_count = container_length(table.get("rows"))
                col_count = container_length(table.get("columns"))
                value_count = container_length(table.get("values"))
                present = 1.0
            elif isinstance(table, str) and table.strip():
                row_count = max(1, len(table.splitlines()))
                col_count = max(1, table.count("|") // max(1, row_count))
                value_count = len(table)
                present = 1.0
            else:
                row_count = col_count = value_count = 0
                present = 0.0
            rows.append(
                to_float_tensor(
                    [
                        float(row_count),
                        float(col_count),
                        float(value_count),
                        present,
                    ]
                )
            )
        return stack_feature_matrix(rows)
