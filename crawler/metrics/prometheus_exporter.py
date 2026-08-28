"""Minimal Prometheus text-format exporter for crawl metrics snapshots.

Implements the Prometheus text exposition format with the standard library
only.  The exporter is opt-in via ``metrics.prometheus_enabled`` and is shut
down deterministically through the runtime shutdown manager.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_METRIC_NAME_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def _safe_metric_name(name: str) -> str:
    return "".join(
        character if character in _METRIC_NAME_CHARACTERS else "_"
        for character in name
    )


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise ValueError("metrics snapshot values must be numeric")


def _make_metrics_handler(
    exporter: "PrometheusExporter",
) -> type[BaseHTTPRequestHandler]:
    class MetricsHandler(BaseHTTPRequestHandler):
        """Serve the latest snapshot for ``GET /metrics`` only."""

        def do_GET(self) -> None:
            if self.path != "/metrics":
                self.send_error(404)
                return
            body = exporter.render().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", _CONTENT_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    return MetricsHandler


class PrometheusExporter:
    """Serve the latest crawl metrics snapshot in Prometheus text format."""

    def __init__(self, *, port: int) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port = port
        self._snapshot: dict[str, str] = {}
        self._lock = threading.Lock()
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 0 <= port <= 65_535
        ):
            raise ValueError("prometheus port must be an integer in 0..65535")

    @property
    def port(self) -> int:
        """Return the bound port (the configured port unless 0 was chosen)."""

        if self._server is not None:
            return int(self._server.server_address[1])
        return self._port

    def publish(self, fields: Mapping[str, object]) -> None:
        """Atomically replace the exported snapshot with numeric fields."""

        numeric: dict[str, str] = {}
        for name, value in fields.items():
            if not isinstance(value, (bool, int, float)):
                continue
            try:
                rendered = _format_value(value)
            except ValueError:
                continue
            numeric[_safe_metric_name(str(name))] = rendered
        with self._lock:
            self._snapshot = numeric

    def render(self) -> str:
        """Render the latest snapshot in Prometheus text exposition format."""

        with self._lock:
            snapshot = dict(self._snapshot)
        lines = [
            "# HELP crawl_metrics_snapshot latest crawl metrics snapshot",
            "# TYPE crawl_metrics_snapshot gauge",
        ]
        lines.extend(
            f"{name} {value}" for name, value in sorted(snapshot.items())
        )
        return "\n".join(lines) + "\n"

    def start(self) -> None:
        """Bind and serve ``/metrics`` on a daemon background thread."""

        if self._server is not None:
            raise RuntimeError("prometheus exporter is already running")
        server = ThreadingHTTPServer(
            ("0.0.0.0", self._port),
            _make_metrics_handler(self),
        )
        thread = threading.Thread(
            target=server.serve_forever,
            name="prometheus-exporter",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread

    def stop(self) -> None:
        """Stop serving and release the bound port."""

        server, self._server = self._server, None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()

    async def aclose(self) -> None:
        """Async close hook used by the runtime shutdown manager."""

        self.stop()

    def __del__(self) -> None:
        if self._server is not None:
            try:
                self._server.server_close()
            except OSError:
                pass


__all__ = ["PrometheusExporter"]
