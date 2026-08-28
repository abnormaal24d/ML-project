"""Handler composition package.

This package contains the modular composition of crawl task handlers
organized by media kind and shared infrastructure.
"""

from orchestration.composition.runtime.handler_composition.registry import (
    build_handler_registry,
    build_analysis_router,
)
from orchestration.composition.runtime.handler_composition.page import (
    build_page_handler,
)
from orchestration.composition.runtime.handler_composition.feed import (
    build_feed_handler,
)
from orchestration.composition.runtime.handler_composition.image import (
    build_image_handler,
)
from orchestration.composition.runtime.handler_composition.audio import (
    build_audio_handler,
)
from orchestration.composition.runtime.handler_composition.video import (
    build_video_handler,
)
from orchestration.composition.runtime.handler_composition.document import (
    build_document_handler,
)

__all__ = [
    "build_handler_registry",
    "build_analysis_router",
    "build_page_handler",
    "build_feed_handler",
    "build_image_handler",
    "build_audio_handler",
    "build_video_handler",
    "build_document_handler",
]