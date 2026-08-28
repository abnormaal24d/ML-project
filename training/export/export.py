"""Export trained multimodal model artifacts for inference handoff."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import nn

from config.path_resolution.resolved_config_paths import format_project_path
from multimodal.model.contracts import CollatedBatch

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings


class _TorchscriptArtifact(Protocol):
    def save(self, path: str) -> None: ...


class _TorchscriptTracer(Protocol):
    def __call__(
        self,
        module: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        *,
        _strict: bool,
    ) -> _TorchscriptArtifact: ...


def export_model(
    *,
    model: nn.Module,
    export_directory: Path,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    dataset_root: Path,
    generated_at: datetime,
) -> dict[str, str]:
    """Write model export artifacts next to a completed checkpoint."""

    export_directory.mkdir(parents=True, exist_ok=True)

    was_training = bool(model.training)
    model.train(False)

    try:
        exports: dict[str, str] = {}

        tokenizer_sha256 = _require_tokenizer_sha256(
            training_settings=training_settings
        )
        model_card_path = export_directory / "model_card.md"
        model_card_path.write_text(
            _model_card(
                model=model,
                model_settings=model_settings,
                training_settings=training_settings,
                dataset_root=dataset_root,
                generated_at=generated_at,
            ),
            encoding="utf-8",
        )
        exports["model_card"] = format_project_path(model_card_path)
        exports["inference_config"] = _write_json(
            path=export_directory / "inference_config.json",
            payload={
                "model_settings": model_settings.model_dump(mode="json"),
                "training_backend": training_settings.training_backend,
                "dataset_root": format_project_path(dataset_root),
                "artifact_version": model_settings.artifact_version,
                "model_family": model_settings.model_family,
                "output_modalities": model_settings.output_modalities,
                "generation": model_settings.generation.model_dump(
                    mode="json",
                ),
                "runtime": model_settings.runtime.model_dump(mode="json"),
                "tokenizer_path": "tokenizer/tokenizer.json",
                "tokenizer_sha256": tokenizer_sha256,
                "text_vocab_size": model_settings.raw_text_vocab_size,
                "audio_codec": {
                    "name": "encodec",
                    "sample_rate": model_settings.audio_tokenizer.sample_rate,
                    "n_codebooks": model_settings.audio_tokenizer.n_codebooks,
                    "codebook_size": (
                        model_settings.audio_tokenizer.codebook_size
                    ),
                    "bandwidth": 6.0,
                },
                "video_tokenizer": {
                    "name": "video_tokenizer_v1",
                    # Schema identifier, not credential material.
                    "token_schema": "video_tokens_t_h_w_v1",  # nosec B105
                    "vocab_size": (
                        model_settings.video_generator.video_token_vocab_size
                    ),
                    "frames": model_settings.video_generator.frames,
                    "grid_height": model_settings.video_generator.grid_height,
                    "grid_width": model_settings.video_generator.grid_width,
                    "fps": 8,
                    "height": model_settings.video_generator.resolution,
                    "width": model_settings.video_generator.resolution,
                },
            },
        )
        exports["preprocessing_config"] = _write_json(
            path=export_directory / "preprocessing_config.json",
            payload={
                "text_normalization": "utf8_structure_preserving",
                "path_rules": "project_relative",
                "pii_rules": "quarantine_on_detection",
            },
        )

        generation_heads = (
            ("sequence",)
            if training_settings.training_backend == "dense_transformer"
            else (
                "sequence",
                "generated_image",
                "audio_token",
                "video_generation",
            )
        )
        if training_settings.training_backend == "dense_transformer":
            exports["dense_inference_contract"] = _write_json(
                path=export_directory / "dense_inference_contract.json",
                payload={
                    "schema_version": "dense_inference.v1",
                    "inputs": {
                        "decoder_input_ids": "int64[B,T]",
                        "decoder_attention_mask": "bool[B,T]",
                        "text": "float[B,L_text,D_text] or float[B,D_text]",
                        "image": "float[B,L_image,D_image] or float[B,D_image]",
                        "audio": "float[B,L_audio,D_audio] or float[B,D_audio]",
                        "video": "float[B,L_video,D_video] or float[B,D_video]",
                        "modality_mask": "bool[B,5]",
                    },
                    "outputs": {"sequence_logits": "float[B,T,V]"},
                    "prompt_contract": "prompt ends with <assistant>",
                    "generation_contract": "model emits answer tokens followed by <eos>",
                },
            )

        exports.update(
            _export_torchscript(
                model=model,
                export_directory=export_directory,
                model_settings=model_settings,
                output_heads=generation_heads,
                training_backend=training_settings.training_backend,
            )
        )
        exports.update(
            _export_onnx(
                model=model,
                export_directory=export_directory,
                model_settings=model_settings,
                output_heads=generation_heads,
                training_backend=training_settings.training_backend,
            )
        )
        exports.update(
            _export_safetensors(
                model=model,
                export_directory=export_directory,
            )
        )

        return exports
    finally:
        model.train(was_training)


class _ExportAdapter(nn.Module):
    """Export adapter that exposes multimodal generation outputs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        output_heads: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.model = model
        self.output_heads = output_heads

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch = CollatedBatch(
            sample_ids=["export"],
            text=text,
            image=image,
            audio=audio,
            video=video,
            modality_mask=modality_mask,
            labels=None,
            document=text,
            output_modalities=[("text", "image", "audio", "video")],
            document_mask=(
                modality_mask[:, 1] if modality_mask.shape[1] > 1 else None
            ),
        )

        output = self.model(batch, output_heads=self.output_heads)
        return _select_export_tensors(
            output=output,
            output_heads=self.output_heads,
        )


class _DenseExportAdapter(nn.Module):
    """Export adapter for the token-based causal multimodal decoder."""

    def __init__(self, *, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        modality_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(decoder_input_ids.shape[0])
        batch = CollatedBatch(
            sample_ids=["export"] * batch_size,
            text=text,
            image=image,
            audio=audio,
            video=video,
            modality_mask=modality_mask,
            labels=None,
            document=text,
            output_modalities=[("text",)] * batch_size,
            document_mask=(
                modality_mask[:, 1] if modality_mask.shape[1] > 1 else None
            ),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            prompt_token_count=[int(decoder_input_ids.shape[1])] * batch_size,
        )
        output = self.model(batch, output_heads=("sequence",))
        logits = output.get("sequence_logits")
        if not torch.is_tensor(logits):
            raise KeyError("dense export requires sequence_logits")
        return logits


def _export_torchscript(
    *,
    model: nn.Module,
    export_directory: Path,
    model_settings: ModelSettings,
    output_heads: tuple[str, ...] | None = None,
    training_backend: str = "pipeline_smoke",
) -> dict[str, str]:
    output_path = export_directory / "model.torchscript.pt"
    status_path = export_directory / "torchscript_export_status.json"

    if output_heads is None:
        output_heads = (
            "sequence",
            "generated_image",
            "audio_token",
            "video_generation",
        )

    try:
        export_adapter: nn.Module = (
            _DenseExportAdapter(model=model)
            if training_backend == "dense_transformer"
            else _ExportAdapter(model=model, output_heads=output_heads)
        )
        export_adapter.train(False)

        example = (
            _example_dense_inputs(model_settings=model_settings)
            if training_backend == "dense_transformer"
            else _example_inputs(model_settings=model_settings)
        )

        with torch.no_grad():
            trace = cast(_TorchscriptTracer, torch.jit.trace)
            traced = trace(export_adapter, example, strict=False)

        traced.save(str(output_path))
        _write_json(
            path=status_path,
            payload={
                "status": "ok",
                "sha256": _sha256_hex(output_path),
            },
        )
        return {"torchscript": format_project_path(output_path)}
    except (
        AttributeError,
        ImportError,
        KeyError,
        RuntimeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - depends on torch runtime.
        _write_json(
            path=status_path,
            payload={"status": "skipped", "reason": str(exc)},
        )
        return {"torchscript_status": format_project_path(status_path)}


def _export_safetensors(
    *,
    model: nn.Module,
    export_directory: Path,
) -> dict[str, str]:
    output_path = export_directory / "model.safetensors"
    status_path = export_directory / "safetensors_export_status.json"

    try:
        from safetensors.torch import save_file

        state_dict = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
            if torch.is_tensor(tensor)
        }

        save_file(state_dict, str(output_path))
        _write_json(
            path=status_path,
            payload={
                "status": "ok",
                "sha256": _sha256_hex(output_path),
            },
        )
        return {"safetensors": format_project_path(output_path)}
    except (
        ImportError,
        RuntimeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - depends on optional safetensors runtime.
        _write_json(
            path=status_path,
            payload={"status": "skipped", "reason": str(exc)},
        )
        return {"safetensors_status": format_project_path(status_path)}


def _export_onnx(
    *,
    model: nn.Module,
    export_directory: Path,
    model_settings: ModelSettings,
    output_heads: tuple[str, ...] | None = None,
    training_backend: str = "pipeline_smoke",
) -> dict[str, str]:
    output_path = export_directory / "model.onnx"
    status_path = export_directory / "onnx_export_status.json"

    if output_heads is None:
        output_heads = (
            "sequence",
            "generated_image",
            "audio_token",
            "video_generation",
        )

    try:
        export_adapter: nn.Module = (
            _DenseExportAdapter(model=model)
            if training_backend == "dense_transformer"
            else _ExportAdapter(model=model, output_heads=output_heads)
        )
        export_adapter.train(False)

        if training_backend == "dense_transformer":
            output_names = ["sequence_logits"]
            example_inputs = _example_dense_inputs(
                model_settings=model_settings
            )
            input_names = [
                "text",
                "image",
                "audio",
                "video",
                "modality_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            dynamic_axes = {
                "text": {0: "batch"},
                "image": {0: "batch"},
                "audio": {0: "batch"},
                "video": {0: "batch"},
                "modality_mask": {0: "batch"},
                "decoder_input_ids": {0: "batch", 1: "sequence"},
                "decoder_attention_mask": {0: "batch", 1: "sequence"},
                "sequence_logits": {0: "batch", 1: "sequence"},
            }
        else:
            output_names = [
                f"{head}_logits"
                if head != "generated_image"
                else "generated_image"
                for head in output_heads
            ]
            example_inputs = _example_inputs(model_settings=model_settings)
            input_names = [
                "text",
                "image",
                "audio",
                "video",
                "modality_mask",
            ]
            dynamic_axes = {
                "text": {0: "batch"},
                "image": {0: "batch"},
                "audio": {0: "batch"},
                "video": {0: "batch"},
                "modality_mask": {0: "batch"},
                **{name: {0: "batch"} for name in output_names},
            }

        torch.onnx.export(
            export_adapter,
            example_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
        _write_json(
            path=status_path,
            payload={
                "status": "ok",
                "sha256": _sha256_hex(output_path),
            },
        )
        return {"onnx": format_project_path(output_path)}
    except (
        AttributeError,
        ImportError,
        KeyError,
        RuntimeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - optional exporter runtime.
        _write_json(
            path=status_path,
            payload={"status": "skipped", "reason": str(exc)},
        )
        return {"onnx_status": format_project_path(status_path)}


def _example_inputs(
    *,
    model_settings: ModelSettings,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    feature_dimensions = model_settings.feature_dimensions

    return (
        torch.zeros(1, int(feature_dimensions["text"])),
        torch.zeros(1, int(feature_dimensions["image"])),
        torch.zeros(1, int(feature_dimensions["audio"])),
        torch.zeros(1, int(feature_dimensions["video"])),
        torch.ones(1, 5, dtype=torch.bool),
    )


def _example_dense_inputs(
    *, model_settings: ModelSettings
) -> tuple[torch.Tensor, ...]:
    base = _example_inputs(model_settings=model_settings)
    prompt_length = min(8, model_settings.text_decoder.max_target_tokens)
    decoder_input_ids = torch.full((1, prompt_length), 2, dtype=torch.long)
    decoder_attention_mask = torch.ones((1, prompt_length), dtype=torch.bool)
    return (*base, decoder_input_ids, decoder_attention_mask)


_EXPORT_OUTPUT_KEYS = {
    "sequence": ("sequence_logits", "text_logits"),
    "generated_image": ("generated_image", "image_logits"),
    "audio_token": ("audio_token_logits", "audio_logits"),
    "video_generation": ("video_token_logits", "video_logits"),
    "classifier": ("label_logits", "logits"),
}


def _select_export_tensors(
    *,
    output: object,
    output_heads: tuple[str, ...],
) -> tuple[torch.Tensor, ...]:
    if torch.is_tensor(output):
        return (output,)

    if not isinstance(output, dict):
        raise TypeError(
            "export adapter expected model output to be a tensor or dict"
        )

    tensors: list[torch.Tensor] = []

    for head in output_heads:
        for key in _EXPORT_OUTPUT_KEYS.get(head, (head,)):
            value = output.get(key)
            if torch.is_tensor(value):
                tensors.append(value)
                break

    if not tensors:
        raise KeyError(
            f"model output did not contain export tensors for {output_heads}"
        )

    return tuple(tensors)


def _write_json(*, path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return format_project_path(path)


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MODEL_CARD_PLACEHOLDERS = (
    "<release-id>",
    "<candidate|production>",
    "<sha256>",
    "<sha256>",
    "<snapshot-id>",
    "<sha256>",
    "<sha256>",
    "<sha256>",
    "<sha256>",
    "<artifact or acceptance-report reference>",
    "<artifact or acceptance-report reference>",
    "<artifact or acceptance-report reference>",
    "<artifact or acceptance-report reference>",
    "<name, version, revision, content hash>",
    "<manifest/lineage reference>",
    "<supported users, tasks, modalities, and operating context>",
    (
        "<unsupported tasks/modalities, disabled experimental routes, "
        "and prohibited uses>"
    ),
    "<reference>",
    "<reference>",
    "<reference>",
    "<reference>",
    "<reference>",
    "<reference>",
    "<reference>",
    "<reference>",
    "<reference or not available>",
    (
        "<known failure modes, dataset limits, modality gaps, "
        "fairness/safety risks, and rollback trigger>"
    ),
    "<rejected|candidate|production_model>",
    "<role/name>",
    "<UTC timestamp>",
    "<none, or explicit non-production blockers>",
)


def _model_card(
    *,
    model: nn.Module,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    dataset_root: Path,
    generated_at: datetime,
) -> str:
    """Render the normative release model-card template with real evidence."""

    template = _load_model_card_template()
    observed_placeholders = tuple(re.findall(r"<[^>]+>", template))
    if observed_placeholders != _MODEL_CARD_PLACEHOLDERS:
        raise RuntimeError(
            "training/export/model_card_template.md placeholders changed "
            "without an export renderer update"
        )
    template = template.replace(
        "# MultimodalModel release model card template",
        "# MultimodalModel release model card",
        1,
    )
    template = template.replace(
        "Use this template for every candidate and production release. "
        "Replace all angle-\n"
        "bracket placeholders; do not remove required sections when a value "
        "is unknown.\n"
        "Record an explicit `not available` plus the blocking gate instead.\n\n",
        "",
        1,
    )

    tokenizer_sha256 = _require_tokenizer_sha256(
        training_settings=training_settings
    )
    configuration_sha256 = _json_sha256(
        {
            "model_settings": model_settings.model_dump(mode="json"),
            "training_settings": training_settings.model_dump(mode="json"),
        }
    )
    initialization_sha256 = _json_sha256(
        {
            "artifact_version": model_settings.artifact_version,
            "model_family": model_settings.model_family,
            "seed": training_settings.seed,
            "schema": "scratch_xavier_v1",
        }
    )
    model_state_sha256 = _model_state_sha256(model=model)
    code_sha256 = _model_code_sha256(model=model)
    release_id = (
        f"{model_settings.artifact_version}-"
        f"{training_settings.release_stage}-"
        f"{model_state_sha256[:12]}"
    )
    card_stage = (
        "production"
        if training_settings.release_stage == "production_model"
        else "candidate"
    )
    decision = (
        "production_model"
        if training_settings.release_stage == "production_model"
        else "candidate"
    )
    tasks = ", ".join(training_settings.tasks) or "all configured tasks"
    modalities = ", ".join(model_settings.enabled_modalities)
    dataset_reference = format_project_path(dataset_root)
    generated_at_text = _utc_isoformat(generated_at)

    replacements = iter(
        (
            release_id,
            card_stage,
            code_sha256,
            configuration_sha256,
            dataset_root.name or dataset_reference,
            tokenizer_sha256,
            initialization_sha256,
            (
                "not available — blocking gate: initialization evidence in "
                "evaluation/acceptance_report.json"
            ),
            (
                f"sha256:{model_state_sha256} (selected in-memory model "
                "state exported from the best checkpoint)"
            ),
            "evaluation/acceptance_report.json#initialization",
            "checkpoint manifest plus SHA-256 sidecar",
            "evaluation/reproducibility_report.json",
            "evaluation/acceptance_report.json#architecture-boundary",
            (
                f"none; training_backend={training_settings.training_backend}; "
                f"tokenizer={training_settings.text_tokenizer_name}; "
                f"tokenizer_sha256={tokenizer_sha256}"
            ),
            (
                f"{dataset_reference}; training_manifest.json; "
                "evaluation/release_evidence_bundle.json"
            ),
            (
                "Project release operators and approved inference services; "
                f"tasks={tasks}; modalities={modalities}; offline, "
                "configuration-pinned inference using the exported artifacts."
            ),
            (
                "Online training, remote model calls, external production "
                "judges, unapproved tasks, unsupported modalities, and use "
                "outside the validated dataset and runtime envelope are "
                "out of scope."
            ),
            f"{dataset_reference} (finalized dataset snapshot root)",
            "training_manifest.json#payload-lineage-reconciliation",
            "training_manifest.json#preprocessing-provenance",
            "evaluation/leakage_report.json",
            "training_metrics.json and evaluation/acceptance_report.json",
            "evaluation/acceptance_report.json#slice-metrics",
            "evaluation/reproducibility_report.json",
            "evaluation/acceptance_report.json#security-compliance-sbom",
            (
                "evaluation/acceptance_report.json; written only after the "
                "independent production gate completes"
            ),
            (
                "Known risks include distribution shift, sparse task or "
                "modality slices, multimodal alignment failures, generation "
                "errors, and residual fairness or safety gaps. Roll back when "
                "acceptance thresholds, integrity checks, latency or memory "
                "limits, or monitored safety gates regress."
            ),
            decision,
            "automated training export; final decision owner: release operator",
            generated_at_text,
            (
                "Initial-state fingerprint and final acceptance decision remain "
                "gated by the reproducibility and acceptance reports; the "
                "exporter does not self-approve a production release."
            ),
        )
    )
    rendered = re.sub(r"<[^>]+>", lambda _match: next(replacements), template)
    try:
        next(replacements)
    except StopIteration:
        pass
    else:  # pragma: no cover - guarded by the placeholder contract above.
        raise RuntimeError("unused model-card template replacement")
    if re.search(r"<[^>]+>", rendered):
        raise RuntimeError(
            "model-card template contains unresolved placeholders"
        )
    return rendered.rstrip() + "\n"


def _utc_isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("model export generated_at must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _load_model_card_template() -> str:
    template_path = Path(__file__).with_name("model_card_template.md")
    try:
        return template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "unable to load packaged model-card template: "
            f"{template_path}: {type(exc).__name__}"
        ) from exc


def _require_tokenizer_sha256(
    *,
    training_settings: TrainingSettings,
) -> str:
    tokenizer_sha256 = training_settings.text_tokenizer_sha256
    if tokenizer_sha256 is None:
        raise ValueError(
            "text_tokenizer_sha256 is required before model export"
        )
    return tokenizer_sha256


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_code_sha256(*, model: nn.Module) -> str:
    model_type = type(model)
    source_path = inspect.getsourcefile(model_type)
    if source_path is not None:
        path = Path(source_path)
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            pass
    try:
        source = inspect.getsource(model_type)
    except (OSError, TypeError):
        source = f"{model_type.__module__}.{model_type.__qualname__}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _model_state_sha256(*, model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        tensor = tensor.contiguous().clone()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
        digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()
