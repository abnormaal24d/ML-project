import hashlib
import json

from training.runtime.checkpoint.io import resolve_checkpoint_model_path


def test_uppercase_sha_manifest(tmp_path):
    # prepare blobstore and model
    root = tmp_path / "blobstore"
    root.mkdir()
    content = b"hello-checkpoint"
    sha = hashlib.sha256(content).hexdigest()
    sha_upper = sha.upper()
    target_dir = root / sha[:2] / sha
    target_dir.mkdir(parents=True)
    model_path = target_dir / "checkpoint.pt"
    model_path.write_bytes(content)
    # write checksum sidecar
    checksum_path = target_dir / "checkpoint.pt.sha256"
    checksum_path.write_text(f"{sha}  checkpoint.pt\n", encoding="ascii")

    # create manifest with uppercase sha
    manifest = {
        "schema_version": 1,
        "kind": "blob",
        "blob_storage": str(root),
        "file": "checkpoint.pt",
        "sha256": sha_upper,
    }
    manifest_path = tmp_path / "checkpoint.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved = resolve_checkpoint_model_path(manifest_path)
    assert resolved is not None
    assert resolved == model_path.resolve()
