import hashlib
from pathlib import Path

from mmcrawler_datasets.training_samples.artifact_path import (
    ValidatedArtifactPath,
)
from mmcrawler_datasets.training_samples.models import _build_object
from preprocessing.privacy.clearance import ApprovedObjectRole


def test_build_object_calculates_exact_sha256(tmp_path: Path):
    path = tmp_path / "object.bin"
    path.write_bytes(b"object bytes")
    obj = _build_object(
        object_id="o1",
        object_path=ValidatedArtifactPath(
            relative_path="object.bin",
            resolved_path=path,
            project_root=tmp_path,
        ),
        object_mime_type="application/octet-stream",
        role=ApprovedObjectRole.PRIMARY_MEDIA,
    )
    assert obj is not None
    assert obj.object_sha256 == hashlib.sha256(b"object bytes").hexdigest()
