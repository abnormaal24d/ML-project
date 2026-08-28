# Task Maturity Matrix

> **Generated artefact.** Do not edit by hand.
> Source of truth: `multimodal.tasks.TASKS`.
> Regenerate with: `python -m scripts.generate_task_matrix`

## Purpose

This matrix is derived from the single canonical task registry.
It is documentation only; runtime governance reads `TaskDefinition`
properties (`maturity`, `sensitivity`, `production_blocked`,
`required_approvals`).

## Maturity and sensitivity

| Maturity / sensitivity | Production | Approvals |
| --- | --- | --- |
| stable + standard | allowed | none |
| beta + standard | conditional | beta |
| beta + sensitive | conditional | beta + sensitive |
| experimental | blocked | none |
| disabled | blocked | none |
| biometric_identity | blocked | none |

## Registry (57 tasks)

| Task | Family | Inputs | Outputs | Maturity | Sensitivity | Sample source | Production | Approvals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `accessibility_audit` | `screen_ui` | image, layout, text | text, json | `experimental` | `standard` | `external` | blocked | — |
| `action_recognition` | `video` | video | class | `beta` | `standard` | `external` | conditional | beta |
| `audio_emotion` | `audio` | audio, text | class | `beta` | `sensitive` | `external` | conditional | beta, sensitive |
| `audio_summarization` | `audio` | audio, text | text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `audio_text_pair` | `audio` | audio, text | class | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `background_sound_classification` | `audio` | audio | class | `experimental` | `standard` | `external` | blocked | — |
| `chart_generation` | `table_spreadsheet` | text | json | `experimental` | `standard` | `external` | blocked | — |
| `chart_qa` | `image` | image, text | text, json | `experimental` | `standard` | `external` | blocked | — |
| `classification` | `text` | text | class | `stable` | `standard` | `external` | allowed | — |
| `cross_modal_consistency` | `cross_modal` | text | class | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `data_anomaly_detection` | `table_spreadsheet` | document, text | class, text | `experimental` | `standard` | `external` | blocked | — |
| `doc_qa` | `document` | document, text | text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `document_comparison` | `document` | document, text | text, json, class | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `document_summarization` | `document` | document, text | text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `document_text_pair` | `document` | document, text | class | `stable` | `standard` | `crawler_derived` | allowed | — |
| `duplicate_detection` | `retrieval` | text | class | `experimental` | `standard` | `external` | blocked | — |
| `image_classification` | `image` | image | class | `experimental` | `standard` | `external` | blocked | — |
| `image_editing` | `image` | image, text, mask | image | `disabled` | `standard` | `external` | blocked | — |
| `image_text_pair` | `image` | image, text | class | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `information_extraction` | `text` | text | text, json | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `instruction_following` | `text` | text | text | `beta` | `standard` | `external` | conditional | beta |
| `meme_explanation` | `image` | image, text | text | `experimental` | `standard` | `external` | blocked | — |
| `multifile_reasoning` | `multimodal_reasoning` | document, text | text, json | `experimental` | `standard` | `external` | blocked | — |
| `multimodal_retrieval` | `cross_modal` | text | class, text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `object_recognition` | `image` | image | json, class | `experimental` | `standard` | `external` | blocked | — |
| `ocr_parse` | `image` | image, layout | text, json | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `passage_retrieval` | `retrieval` | document, text | class, text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `pdf_text_pair` | `document` | document, text | class | `stable` | `standard` | `crawler_derived` | allowed | — |
| `representation` | `text` | text | class | `stable` | `standard` | `self_supervised` | allowed | — |
| `scene_retrieval` | `retrieval` | video, text | class, text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `scene_understanding` | `video` | video | json | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `screen_to_steps` | `screen_ui` | image, text | text, json | `experimental` | `standard` | `external` | blocked | — |
| `screenshot_qa` | `screen_ui` | image, text | text | `experimental` | `standard` | `external` | blocked | — |
| `semantic_search` | `retrieval` | text | class, text | `beta` | `standard` | `external` | conditional | beta |
| `speaker_diarization` | `audio` | audio | json | `disabled` | `sensitive` | `external` | blocked | — |
| `speaker_id` | `audio` | audio | class | `disabled` | `biometric_identity` | `external` | blocked | — |
| `speech_reconstruction` | `audio` | audio | audio | `experimental` | `standard` | `external` | blocked | — |
| `speech_to_audio` | `audio` | text | audio | `disabled` | `standard` | `external` | blocked | — |
| `speech_transcription` | `audio` | audio | text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `speech_translation` | `audio` | audio | text, audio | `beta` | `standard` | `external` | conditional | beta |
| `spreadsheet_analysis` | `table_spreadsheet` | document, text | text, json | `experimental` | `standard` | `external` | blocked | — |
| `summarization` | `text` | text | text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `table_extraction` | `document` | document, text, layout | json | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `table_qa` | `table_spreadsheet` | document, text | text | `experimental` | `standard` | `external` | blocked | — |
| `text_pretrain` | `text` | text | text | `stable` | `standard` | `self_supervised` | allowed | — |
| `text_to_image` | `image` | text | image | `disabled` | `standard` | `external` | blocked | — |
| `text_to_video` | `video` | text | video | `disabled` | `standard` | `external` | blocked | — |
| `translation` | `text` | text | text | `experimental` | `standard` | `external` | blocked | — |
| `ui_error_diagnosis` | `screen_ui` | image, text | text, json | `experimental` | `standard` | `external` | blocked | — |
| `ui_to_code` | `screen_ui` | image, layout, text | code | `experimental` | `standard` | `external` | blocked | — |
| `video_captioning` | `video` | video | text | `experimental` | `standard` | `crawler_derived` | blocked | — |
| `video_editing` | `video` | video, text, mask | video | `disabled` | `standard` | `external` | blocked | — |
| `video_qa` | `video` | video, text | text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `video_summarization` | `video` | video, text | text | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `video_text_pair` | `video` | video, text | class | `beta` | `standard` | `crawler_derived` | conditional | beta |
| `visual_math` | `image` | image | text | `experimental` | `standard` | `external` | blocked | — |
| `vqa` | `image` | image, text | text | `beta` | `standard` | `crawler_derived` | conditional | beta |

## Invariants

- every task in `TASKS` appears here;
- configuration may only reference names present in `TASKS`;
- routing inputs/outputs resolve from `TASKS` plus optional overrides;
- governance is derived from `maturity` and `sensitivity` only.
