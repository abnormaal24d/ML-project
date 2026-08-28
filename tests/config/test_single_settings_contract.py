"""The settings tree exposes exactly one training and runtime contract.

The profile-based root uses the canonical multimodal schemas directly
instead of subclassing them with different defaults, so ``TrainingSettings()``
and ``RuntimeSettings()`` produce the same model no matter which import path
is used.
"""

from __future__ import annotations

import config.multimodal.fusion_settings as multimodal_fusion
import config.multimodal.training_settings as multimodal_training
from config.multimodal.model_settings import ModelSettings


def test_training_settings_is_single_shared_class() -> None:
    assert multimodal_training.TrainingSettings is (
        multimodal_training.TrainingSettings
    )


def test_runtime_settings_is_single_shared_class() -> None:
    assert multimodal_fusion.RuntimeSettings is (
        multimodal_fusion.RuntimeSettings
    )


def test_training_defaults_match_consolidated_contract() -> None:
    settings = multimodal_training.TrainingSettings()
    assert settings.text_tokenizer_vocab_size == 4096
    assert settings.dynamic_sampling is True
    assert settings.early_stopping_patience == 2
    assert settings.early_stopping_min_delta == 0.0001
    assert settings.pin_memory is False
    assert settings.progress_log_interval_batches == 10
    assert settings.mlm_loss_weight == 0.25
    assert settings.language_modeling_loss_weight == 0.25
    assert settings.image_patch_loss_weight == 0.0
    assert settings.audio_masked_loss_weight == 0.0
    assert settings.video_temporal_loss_weight == 0.0
    assert settings.hard_negative_loss_weight == 0.0
    assert settings.use_hard_negative_sampler is False


def test_runtime_defaults_match_consolidated_contract() -> None:
    settings = multimodal_fusion.RuntimeSettings()
    assert settings.cache_kv is False
    assert settings.max_batch_tokens == 4096


def test_root_settings_use_same_defaults() -> None:
    from config.settings.root import Settings

    root = Settings(profile="dev")
    assert root.training.text_tokenizer_vocab_size == 4096
    assert root.training.dynamic_sampling is True
    assert root.multimodal.runtime.cache_kv is False
    assert root.multimodal.runtime.max_batch_tokens == 4096


def test_root_multimodal_uses_canonical_model_settings() -> None:
    from config.settings.root import Settings

    assert Settings.model_fields["multimodal"].annotation is ModelSettings
    settings = Settings(profile="dev")
    assert type(settings.multimodal) is ModelSettings
