# ADR-0002: Scratch-only MultimodalModel and External Models Strictly Limited to Preprocessing

## Status

Accepted.

## Context

The project is a data pipeline that crawls, preprocesses (with possible external services like Whisper, Tesseract,
RapidOCR, pyannote for extraction), builds datasets, and trains a baseline multimodal model from scratch.

External models are valuable for high-quality derived data (transcripts, OCR, speaker segments, etc.) during
preprocessing. However, they must never leak into the actual training model:

- No loading of external models (transformers, faster-whisper, etc.) as encoders inside `multimodal/model/`.
- No external embeddings, hidden states, or teacher signals during training.
- No downloads or remote calls during effective training.
- The final checkpoint must contain only parameters from one own randomly-initialized `MultimodalModel`.
- External models must not determine production acceptance.

The repository already has strong "scratch-only" comments and some boundaries (see ADR-0001, config comments,
preprocessing isolation). This ADR makes the rule explicit, single-source-of-truth, and technically enforceable.

No parallel `v2`, `new_model`, or second preprocessing tree is allowed. All work happens in-place on the existing
packages.

## Decision

1. **External models are allowed exclusively for preprocessing and data extraction.**
    - Allowed packages/roots: `preprocessing/` (including `preprocessing/media/*`,
      `preprocessing/multimodal_preprocessor.py`).
    - Their outputs are treated as **derived dataset data** with full provenance (producer, revision, hash, parameters,
      confidence, etc.).
    - They must be pinned (name + revision + content hash) in production configs.

2. **The training model is strictly scratch-only.**
    - Single top-level class: `multimodal/model/model.py:MultimodalModel`.
    - All encoders (`Auto*Encoder`, scratch encoders, `LayoutAwareDocumentEncoder`), fusion (`GatedFusion`), heads, etc.
      live under `multimodal/model/` and `training/`.
    - `multimodal/model/`, `training/runtime/`, `training/losses/`, `training/evaluation/`, `mmcrawler_datasets/` (
      except collators for raw inputs) **must never** import or reference external pretrained model stacks.
    - Training must be able to run fully offline (`--network none` in CI/production) using only own tokenizer, own
      checkpoint resume, and finalized snapshots.

3. **Provenance ends at the snapshot.**
    - Preprocessing provenance (including external model details) is recorded in manifests/lineage.
    - From the finalized training snapshot onward, only own `MultimodalModel` parameters exist.

4. **Preprocessing output is dataset data, never an online model connection.**
    - OCR text, transcripts, speaker segments, timestamps, confidence values, and
      other extracted observations may enter a finalized snapshot with complete
      producer provenance.
    - External embeddings, hidden states, teacher logits, distillation targets,
      runtime handles, and model weights may not be supplied to the training
      model or checkpoint.

5. **There is one trainable model family.**
    - `MultimodalModel` is the sole public trainable top-level class. Encoders,
      fusion, routers, and heads are components of that class, not independent
      model families.
    - Its parameters are randomly initialized, and a release checkpoint uses
      one coherent `MultimodalModel` namespace. There is no `v2/`,
      `new_multimodal_model.py`, or alternate checkpoint family.

6. **Enforcement mechanisms (this ADR + follow-ups)**
    - New ADR (this document) + reference from README, model card, product_scope.md.
    - Static dependency and source scanning at the preprocessing/training boundary.
    - Dependency split in `pyproject.toml` (preprocessing-media vs training-scratch extras).
    - Docker multistage images.
    - Runtime guard + CI that blocks network + hub access during training.
    - Checkpoint metadata must record pure random init fingerprint.
    - Config validation rejects `pretrained_*`, `encoder_checkpoint`, `hf_model_id` etc. for production training paths.

## Allow/deny matrix

This matrix is normative. An item not explicitly allowed on a trainable path is
denied until this ADR is superseded.

| Boundary | Allowed | Denied |
|---|---|---|
| `preprocessing/`                                                    | Invoke pinned Whisper/faster-whisper, Tesseract, RapidOCR, or pyannote backends for offline extraction; record name, revision, content hash, parameters, confidence, source hash, and output hash. | Train an external backend with `MultimodalModel`; emit embeddings, hidden states, teacher logits, or distillation targets for model input.                      |
| Finalized dataset snapshot and `datachecker/` lineage               | Store extracted observations such as OCR text, transcripts, speaker segments, timestamps, and quality metadata as derived dataset data with complete provenance.                                   | Store executable model connections, remote handles, model caches/weights, or external embeddings/hidden states intended as encoder input.                       |
| `mmcrawler_datasets/`                                               | Read finalized records, tensorize raw or derived observations, sample, and collate.                                                                                                                | Import, invoke, or download an external model; turn external embeddings or hidden states into a training shortcut.                                              |
| `multimodal/model/`                                                 | Define the one own, randomly initialized `MultimodalModel` and improve its encoders, fusion, routers, and heads in place.                                                                          | Load pretrained encoders, model-hub artifacts, per-encoder checkpoints, external embeddings/hidden states, teachers, or a parallel model family.                |
| `training/runtime/`, `training/losses/`, and `evaluator/` | Train, resume, and evaluate the own complete checkpoint offline against finalized snapshots and repository-owned metrics.                                                                          | Download or call an external model, use teacher/distillation signals, make remote inference calls, or delegate the final quality decision to an external judge. |
| Checkpoint and export bundle                                        | Contain one coherent `MultimodalModel` parameter namespace plus the own tokenizer, config, fingerprints, and required evidence.                                                                    | Contain external encoder weights, loose per-encoder checkpoints, preprocessing model caches, remote model identifiers, or an alternate checkpoint family.       |
| Production acceptance                                               | Use deterministic repository gates and leakage-free representative evaluation evidence.                                                                                                             | Let an external judge or preprocessing model determine the production acceptance decision.                                                                      |

## Consequences

- Preprocessing may continue to use (and must properly pin + provenance) Whisper, Tesseract, RapidOCR, pyannote, etc.
- Training image / `training-scratch` extra will never pull `transformers`, `faster-whisper`, etc.
- Any future feature that wants to "use a strong external encoder" must either:
  a) use only extraction observations or labels produced in preprocessing, with
  full provenance and without embeddings/hidden states; or
  b) implement the trainable capability as a randomly initialized component
  inside the existing model packages.
- All existing "scratch-only" comments are now backed by enforceable rules and automated checks.
- Release acceptance will explicitly verify absence of external model artefacts in the training bundle.

## References

- [ADR-0001](0001-package-boundaries.md) (package boundaries)
- [Product scope](../product_scope.md)
- [Model card template](../../training/export/model_card_template.md)
- [Component ownership matrix](../architecture/code_architecture_100_score_gate.md#component-ownership-matrix)
- [Task maturity matrix](../task_maturity_matrix.md) (release-readiness governance)
- Existing scratch-only rules comments in `config/multimodal/*.py`
- `preprocessing/media/*` (current home of external services)
- `multimodal/model/model.py` (sole top-level trainable model)
- Future: points 1-10 and 41-65 of the boundary implementation plan.

## Acceptance

- Documentation review asserts the presence of this ADR, its allow/deny matrix,
  and references from the README and model card template.
- Import/string scanner fails CI on forbidden import or "from_pretrained"/hub string inside `multimodal/model/`,
  `training/runtime/`, `training/evaluation/`.
- `pip install ".[training-scratch]" --constraint requirements/constraints-py312-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu` plus `pip freeze` contains
  none of the external model packages.
- A tiny training run with network disabled succeeds using only own components.
- A production candidate manifest records only own `MultimodalModel` fingerprint (no external encoder weights).

## Related Guardrails

See also the 100-point implementation plan (Fase 1 points 1-10 for boundaries, Fase 2 for provenance in preprocessing).
