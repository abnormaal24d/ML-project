"""Dev NOAA seed expansion must not prefer the blocked www.noaa.gov host."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from config.source_catalog.registry_expansion import apply_source_registry


def test_dev_noaa_seed_subset_avoids_blocked_primary_host() -> None:
    project_root = Path(__file__).resolve().parents[2]
    mapping: dict[str, object] = {
        "sources": {
            "active_profile": "public_science_small",
            "registry_sources": [{"name": "noaa", "max_seeds": 12}],
        }
    }
    apply_source_registry(
        mapping,
        project_root=project_root,
        environment="dev",
        registry_path="config/files/sources/source_registry.json",
    )

    sources = mapping["sources"]
    assert isinstance(sources, dict)
    seed_urls = sources.get("seed_urls")
    assert isinstance(seed_urls, list)
    assert len(seed_urls) == 12
    hosts = {urlsplit(str(url)).hostname for url in seed_urls}
    assert "www.noaa.gov" not in hosts
    assert hosts >= {
        "repository.library.noaa.gov",
        "oceanservice.noaa.gov",
        "oceanexplorer.noaa.gov",
    }


def test_noaa_registry_seed_prefix_excludes_www_noaa() -> None:
    project_root = Path(__file__).resolve().parents[2]
    registry = json.loads(
        (
            project_root
            / "config"
            / "files"
            / "sources"
            / "source_registry.json"
        ).read_text(encoding="utf-8")
    )
    seed_urls = registry["sources"]["noaa"]["seed_urls"]
    first_twelve = seed_urls[:12]
    assert all(
        urlsplit(url).hostname != "www.noaa.gov" for url in first_twelve
    )
    assert {urlsplit(url).hostname for url in first_twelve} >= {
        "repository.library.noaa.gov",
        "oceanservice.noaa.gov",
        "oceanexplorer.noaa.gov",
    }
