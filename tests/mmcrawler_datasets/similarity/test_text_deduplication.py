"""Dataset-level text deduplication is deterministic and reentrant."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from mmcrawler_datasets.similarity.text_deduplication import (
    NearTextDeduplicator,
)


def test_assign_clusters_is_reentrant_across_concurrent_runs() -> None:
    deduper = NearTextDeduplicator(
        threshold=0.75,
        shingle_width=2,
        use_buckets=False,
    )
    corpora = tuple(
        {
            f"run-{run_index}-a": "alpha beta gamma delta epsilon",
            f"run-{run_index}-b": "alpha beta gamma delta changed",
            f"run-{run_index}-c": "completely unrelated source material",
        }
        for run_index in range(24)
    )
    expected = tuple(
        deduper.assign_clusters(texts_by_document_id=corpus)
        for corpus in corpora
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        actual = tuple(
            executor.map(
                lambda corpus: deduper.assign_clusters(
                    texts_by_document_id=corpus
                ),
                corpora,
            )
        )

    assert actual == expected
    for run_index, assignments in enumerate(actual):
        assert set(assignments) == {
            f"run-{run_index}-a",
            f"run-{run_index}-b",
            f"run-{run_index}-c",
        }
