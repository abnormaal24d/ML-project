"""Concrete Pyannote diarization backend adapter."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.environment.runtime_environment import configured_token_value
from config.errors import RuntimeDependencyError

if TYPE_CHECKING:
    from config.preprocessing.media_settings import DiarizationSettings


def validate_diarization_artifact(*, settings: DiarizationSettings) -> None:
    """Validate a local pyannote origin and its optional integrity pin."""

    local_path = settings.local_model_path
    origin = Path(local_path or settings.model_name)
    if settings.local_files_only and not origin.exists():
        raise RuntimeDependencyError(
            "configured pyannote local model path does not exist",
            setting="preprocessing.diarization.local_model_path",
            required_artifact=origin.name or "model_artifact",
            issue="required_local_artifact_missing",
        )
    expected_hash = settings.model_artifact_hash
    if local_path and expected_hash and _path_sha256(origin) != expected_hash:
        raise RuntimeDependencyError(
            "configured pyannote local model artifact hash mismatch",
            setting="preprocessing.diarization.model_artifact_hash",
            required_artifact=origin.name or "model_artifact",
            issue="artifact_hash_mismatch",
        )


def load_pyannote_backend(
    *, settings: DiarizationSettings, device: str
) -> Any:
    """Build the single real pyannote pipeline for the configured backend.

    Ownership: this is the only place that calls ``Pipeline.from_pretrained``.
    Failures raise with the original root cause preserved.
    """

    validate_diarization_artifact(settings=settings)

    try:
        import torch
        from pyannote.audio import Pipeline  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "configured pyannote diarization runtime cannot be imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    model_name = settings.model_name.strip()
    local_path = (
        settings.local_model_path.strip()
        if settings.local_model_path is not None
        else None
    )
    origin = local_path or model_name
    if origin is None:
        raise RuntimeError(
            "configured pyannote backend requires model_name or local_model_path"
        )
    kwargs: dict[str, object] = {}
    signature = inspect.signature(Pipeline.from_pretrained)
    revision = settings.model_revision
    if revision and "revision" in signature.parameters:
        kwargs["revision"] = revision
    elif revision and local_path is None:
        origin = f"{origin}@{revision}"

    if local_path is None:
        token_name = settings.token_environment_variable
        token = configured_token_value(token_name)
        if not token:
            raise RuntimeError(
                "remote pyannote diarization requires the configured token "
                f"environment variable '{token_name}'"
            )
        if "use_auth_token" in signature.parameters:
            kwargs["use_auth_token"] = token
        elif "token" in signature.parameters:
            kwargs["token"] = token

    try:
        pipeline = Pipeline.from_pretrained(origin, **kwargs)
    except Exception as exc:  # noqa: BLE001 — preserve root cause for operators
        raise RuntimeError(
            f"configured pyannote pipeline '{origin}' could not be loaded: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if pipeline is None:
        raise RuntimeError(
            f"configured pyannote pipeline '{origin}' returned no pipeline; "
            "verify model access, accepted user conditions, and token setup"
        )

    pipeline.to(torch.device(device))
    return PyannoteBackend(
        pipeline=pipeline,
        model_version=(settings.backend_version or "pyannote-3.x"),
    )


@dataclass(slots=True)
class PyannoteBackend:
    pipeline: Any
    model_version: str

    def __call__(self, target: object, **kwargs: object) -> object:
        output = self.pipeline(target, **kwargs)
        return getattr(output, "speaker_diarization", output)


def _path_sha256(path: Path) -> str:
    digest = sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        )
    )
    for candidate in files:
        relative = (
            candidate.name
            if path.is_file()
            else candidate.relative_to(path).as_posix()
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
