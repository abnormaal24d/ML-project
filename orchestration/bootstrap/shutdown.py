"""OS signal integration for workflow shutdown."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Sequence
from types import FrameType

if True:
    from logger.project_logger import ProjectLogger

ShutdownCallback = Callable[[], None]


def supported_shutdown_signals() -> tuple[signal.Signals, ...]:
    """Return shutdown signals supported by the current platform."""

    signal_names: Sequence[str] = (
        "SIGHUP",
        "SIGTERM",
        "SIGINT",
        "SIGBREAK",
    )
    supported_signals: list[signal.Signals] = []

    for signal_name in signal_names:
        candidate = getattr(signal, signal_name, None)
        if isinstance(candidate, signal.Signals):
            supported_signals.append(candidate)

    return tuple(dict.fromkeys(supported_signals))


def install_signal_handlers(
    *,
    loop: asyncio.AbstractEventLoop,
    logger: ProjectLogger,
    shutdown_callback_factory: Callable[
        [signal.Signals],
        ShutdownCallback,
    ],
) -> None:
    """Install best-effort shutdown signal handlers."""

    for sig in supported_shutdown_signals():
        callback = shutdown_callback_factory(sig)
        signum = int(sig)

        try:
            loop.add_signal_handler(signum, callback)
            logger.debug(
                "signal_handler_registered",
                signal_name=sig.name,
                strategy="asyncio",
            )
            continue
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            logger.debug(
                "signal_handler_asyncio_registration_failed",
                signal_name=sig.name,
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )

        try:

            def handler(
                _signum: int,
                _frame: FrameType | None,
                _callback: ShutdownCallback = callback,
            ) -> None:
                _callback()

            signal.signal(signum, handler)
            logger.debug(
                "signal_handler_registered",
                signal_name=sig.name,
                strategy="signal_module",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "signal_handler_registration_skipped",
                signal_name=sig.name,
                error_type=type(exc).__name__,
            )
