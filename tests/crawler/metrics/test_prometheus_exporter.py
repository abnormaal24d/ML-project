"""Coverage for the stdlib Prometheus text-format metrics exporter."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from crawler.metrics.prometheus_exporter import PrometheusExporter


def test_render_produces_prometheus_text_format() -> None:
    exporter = PrometheusExporter(port=0)
    exporter.publish(
        {
            "fetch_attempts_total": 42,
            "avg_latency_seconds": 1.5,
            "top_hosts": [{"host": "example.test", "requests": 1}],
            "skip_rate": 0.0,
        }
    )

    rendered = exporter.render()

    assert "# HELP crawl_metrics_snapshot" in rendered
    assert "# TYPE crawl_metrics_snapshot gauge" in rendered
    assert "avg_latency_seconds 1.5" in rendered
    assert "fetch_attempts_total 42" in rendered
    assert "skip_rate 0" in rendered
    assert "top_hosts" not in rendered
    exporter.stop()


def test_render_is_sorted_and_numeric_fields_only() -> None:
    exporter = PrometheusExporter(port=0)
    exporter.publish({"b": 2, "a": 1, "not-a-number": object()})

    assert exporter.render().splitlines()[-1] == "b 2"
    assert "not-a-number" not in exporter.render()
    exporter.stop()


def test_serves_metrics_endpoint_over_http() -> None:
    exporter = PrometheusExporter(port=0)
    exporter.start()
    try:
        exporter.publish({"fetch_attempts_total": 7})
        with urllib.request.urlopen(
            f"http://127.0.0.1:{exporter.port}/metrics",
            timeout=5,
        ) as response:
            assert response.status == 200
            body = response.read().decode("utf-8")
        assert "fetch_attempts_total 7" in body
    finally:
        exporter.stop()


def test_serves_404_for_unknown_paths() -> None:
    exporter = PrometheusExporter(port=0)
    exporter.start()
    try:
        with pytest.raises(Exception) as exc_info:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{exporter.port}/other",
                timeout=5,
            ):
                pass
        assert exc_info.value.code == 404
    finally:
        exporter.stop()


def test_stop_releases_the_bound_port() -> None:
    exporter = PrometheusExporter(port=0)
    exporter.start()
    exporter.stop()
    exporter.start()
    exporter.stop()


def test_aclose_stops_the_server() -> None:
    import asyncio

    exporter = PrometheusExporter(port=0)
    exporter.start()
    asyncio.run(exporter.aclose())
    assert exporter._server is None


def test_invalid_port_is_rejected() -> None:
    with pytest.raises(ValueError):
        PrometheusExporter(port=70_000)


def test_module_is_importable_from_packaged_root() -> None:
    package_root = Path(__file__).resolve().parents[3]
    assert (
        package_root / "crawler" / "metrics" / "prometheus_exporter.py"
    ).is_file()
