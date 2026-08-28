"""Sample-level text-field augmentation for multimodal training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentation.outcomes.augmentation_result import AugmentationRejection
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from augmentation.text.text_augmentation_result import (
    SampleAugmentationResult,
)
from augmentation.text.text_identity import (
    single_line_value,
    text_field_value,
    text_identity,
)
from augmentation.text.text_sample_context import (
    sample_domain,
    sample_modality,
    sample_task_type,
    sample_text_spans,
    sample_title,
)
from augmentation.text.text_variant_planner import (
    plan_augmentation_strategies,
)
from augmentation.text.variants import (
    build_context_variant,
    build_span_focus_variant,
    build_title_variant,
)
from config.environment.default_values import DEFAULT_TRAIN_SPLIT_NAME
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from augmentation.text.text_variant_assembler import (
        TextVariantAssembler,
    )
    from config.augmentation.augmentation_settings import AugmentationSettings
    from mmcrawler_datasets.schema import MultimodalSample


class TextFieldAugmenter:
    """Build deterministic text-field variants for one multimodal sample."""

    def __init__(
        self,
        *,
        settings: AugmentationSettings,
        variant_assembler: TextVariantAssembler,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._variant_assembler = variant_assembler
        self._logger = logger
        self._logger.debug("multimodal_sample_augmenter_initialized")

    def augment(
        self,
        *,
        sample: MultimodalSample,
    ) -> tuple[tuple[str, MultimodalSample], ...]:
        return self.augment_with_rejections(sample=sample).variants

    def augment_with_rejections(
        self,
        *,
        sample: MultimodalSample,
    ) -> SampleAugmentationResult:
        if not self._is_sample_eligible(sample):
            reason = self._build_rejection_reason(sample)
            if reason:
                return SampleAugmentationResult.rejected(
                    self._rejection(sample=sample, reason=reason)
                )
        return self._augment_eligible_sample(sample)

    def _is_sample_eligible(self, sample: MultimodalSample) -> bool:
        if self._disabled_reason() is not None:
            return False
        modality = sample_modality(sample=sample)
        if modality not in self._settings.effective_text_field_modalities:
            return False
        if modality == "document" and not self._settings.document.enabled:
            return False
        base_text = text_field_value(sample.text)
        if len(base_text) < self._settings.text.minimum_text_length:
            return False
        if (
            self._settings.text.maximum_text_length is not None
            and len(base_text) > self._settings.text.maximum_text_length
            and self._settings.text.truncation_rules == "skip"
        ):
            return False
        return True

    def _build_rejection_reason(
        self, sample: MultimodalSample
    ) -> AugmentationRejectionReason | None:
        disabled = self._disabled_reason()
        if disabled:
            return disabled
        modality = sample_modality(sample=sample)
        if modality not in self._settings.effective_text_field_modalities:
            return AugmentationRejectionReason.MODALITY_NOT_ALLOWED
        if modality == "document" and not self._settings.document.enabled:
            return AugmentationRejectionReason.DOCUMENT_AUGMENTATION_DISABLED
        base_text = text_field_value(sample.text)
        if len(base_text) < self._settings.text.minimum_text_length:
            return AugmentationRejectionReason.TEXT_TOO_SHORT
        if (
            self._settings.text.maximum_text_length is not None
            and len(base_text) > self._settings.text.maximum_text_length
            and self._settings.text.truncation_rules == "skip"
        ):
            return AugmentationRejectionReason.TEXT_TOO_LONG
        return None

    def _augment_eligible_sample(
        self,
        sample: MultimodalSample,
    ) -> SampleAugmentationResult:
        metadata = sample.metadata or {}
        title = sample_title(sample=sample, metadata=metadata)
        domain = sample_domain(metadata=metadata)
        task_type = sample_task_type(metadata=metadata)
        text_spans = sample_text_spans(metadata=metadata)
        modality = sample_modality(sample=sample)
        base_text = text_field_value(sample.text)

        variants: list[tuple[str, MultimodalSample]] = []
        rejections: list[AugmentationRejection] = []
        seen_texts = {text_identity(base_text)}

        strategies = plan_augmentation_strategies(
            settings=self._settings,
            source_sample_id=sample.sample_id,
            modality=modality,
            task_type=task_type,
            domain=domain,
        )
        if not strategies:
            return SampleAugmentationResult.rejected(
                self._rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.NO_ENABLED_TEXT_STRATEGY,
                )
            )

        for strategy_name in strategies:
            if len(variants) >= self._settings.text.max_variants_per_sample:
                rejections.append(
                    self._rejection(
                        sample=sample,
                        reason=AugmentationRejectionReason.MAX_VARIANTS_LIMIT_REACHED,
                        variant_name=strategy_name,
                    )
                )
                continue

            variant_text = _build_variant_text(
                strategy_name=strategy_name,
                base_text=base_text,
                title=title,
                modality=modality,
                task_type=task_type,
                domain=domain,
                text_spans=text_spans,
            )
            if variant_text is None:
                rejections.append(
                    self._rejection(
                        sample=sample,
                        reason=_strategy_no_output_reason(
                            strategy_name=strategy_name,
                            title=title,
                            modality=modality,
                            task_type=task_type,
                            domain=domain,
                            text_spans=text_spans,
                        ),
                        variant_name=strategy_name,
                    )
                )
                continue

            rejection_reason = self._append_variant(
                variants=variants,
                seen_texts=seen_texts,
                sample=sample,
                augmentation_name=strategy_name,
                variant_text=variant_text,
            )
            if rejection_reason is not None:
                rejections.append(
                    self._rejection(
                        sample=sample,
                        reason=rejection_reason,
                        variant_name=strategy_name,
                    )
                )

        return SampleAugmentationResult(
            variants=tuple(variants), rejections=tuple(rejections)
        )

    def _disabled_reason(self) -> AugmentationRejectionReason | None:
        if not self._settings.enabled:
            return AugmentationRejectionReason.AUGMENTATION_DISABLED
        if not self._settings.text.enabled:
            return AugmentationRejectionReason.TEXT_AUGMENTATION_DISABLED
        if self._settings.text.max_variants_per_sample == 0:
            return AugmentationRejectionReason.MAX_VARIANTS_ZERO
        return None

    def _append_variant(
        self,
        *,
        variants: list[tuple[str, MultimodalSample]],
        seen_texts: set[str],
        sample: MultimodalSample,
        augmentation_name: str,
        variant_text: str | None,
    ) -> AugmentationRejectionReason | None:
        variant, rejected_reason = self._variant_assembler.build_with_reason(
            sample=sample,
            augmentation_name=augmentation_name,
            variant_text=variant_text,
            seen_texts=seen_texts,
        )
        if variant is None:
            return (
                rejected_reason or AugmentationRejectionReason.VARIANT_REJECTED
            )
        variants.append((augmentation_name, variant))
        return None

    def _rejection(
        self,
        *,
        sample: MultimodalSample,
        reason: AugmentationRejectionReason,
        variant_name: str | None = None,
    ) -> AugmentationRejection:
        metadata = sample.metadata or {}
        return AugmentationRejection(
            reason=reason,
            sample_id=sample.sample_id,
            variant_name=variant_name,
            split=str(metadata.get("split") or DEFAULT_TRAIN_SPLIT_NAME),
            modality=str(
                metadata.get("modality") or sample_modality(sample=sample)
            ),
            task_type=str(metadata.get("task_type") or "unknown"),
            label=sample.label,
            source_url=sample.source_url,
            text_length=len(text_field_value(sample.text)),
        )


def _build_variant_text(
    *,
    strategy_name: str,
    base_text: str,
    title: str | None,
    modality: str | None,
    task_type: str | None,
    domain: str | None,
    text_spans: tuple[str, ...],
) -> str | None:
    if strategy_name == "text_span_focus":
        return build_span_focus_variant(
            text=base_text,
            text_spans=text_spans,
        )
    if strategy_name == "title_prefix":
        return build_title_variant(text=base_text, title=title)
    if strategy_name == "context_prefix":
        return build_context_variant(
            text=base_text,
            modality=modality,
            task_type=task_type,
            domain=domain,
        )
    return None


def _strategy_no_output_reason(
    *,
    strategy_name: str,
    title: str | None,
    modality: str | None,
    task_type: str | None,
    domain: str | None,
    text_spans: tuple[str, ...],
) -> AugmentationRejectionReason:
    if strategy_name == "title_prefix":
        if not title:
            return AugmentationRejectionReason.MISSING_TITLE
        return AugmentationRejectionReason.TITLE_ALREADY_PRESENT
    if strategy_name == "context_prefix":
        if not any(
            _as_opt_str(value) for value in (modality, task_type, domain)
        ):
            return AugmentationRejectionReason.MISSING_CONTEXT
        return AugmentationRejectionReason.CONTEXT_ALREADY_PRESENT
    if strategy_name == "text_span_focus":
        if not text_spans:
            return AugmentationRejectionReason.MISSING_TEXT_SPAN
        return AugmentationRejectionReason.TEXT_SPAN_NOT_APPLICABLE
    return AugmentationRejectionReason.STRATEGY_NO_OUTPUT


def _as_opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = single_line_value(value)
    return text or None
