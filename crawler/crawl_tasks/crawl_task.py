"""Crawl task model, identity, and immutable update semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task_context import CrawlTaskContext
from crawler.crawl_tasks.crawl_task_context_builder import (
    coerce_crawl_task_context,
)
from shared.runtime_primitives import IdGenerator


def generate_task_id(*, id_generator: IdGenerator) -> str:
    """Return a compact process-independent crawl task identifier."""

    raw_id = id_generator.generate()
    normalized = str(raw_id).strip()
    if not normalized:
        raise ValueError("id generator returned an empty task identifier")
    return normalized[:16]


def _clean_optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_required_url(value: str) -> str:
    url = value.strip()
    if not url:
        raise ValueError("crawl task url is required")
    return url


def _clean_required_source_name(value: object) -> str:
    source_name = str(value).strip().lower()
    if not source_name:
        raise ValueError("crawl task source_name is required")
    return source_name


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(
        value, (str, bytes, bytearray, int, float)
    ):
        return None
    try:
        if isinstance(value, str) and "." in value.strip():
            return int(float(value.strip()))
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _strip_url_suffix(url: str) -> str:
    return url.lower().split("?", 1)[0].split("#", 1)[0]


def _guess_seed_kind(url: str) -> MediaKind:
    from crawler.classification.media_kind_registry import (
        match_extension,
    )

    matched_kind = match_extension(_strip_url_suffix(url))
    if matched_kind is not None:
        return matched_kind
    return MediaKind.PAGE


@dataclass(frozen=True, slots=True)
class CrawlTask:
    """Normalized immutable crawl task."""

    url: str
    source_name: str
    task_id: str | None = None
    kind: MediaKind = MediaKind.PAGE
    depth: int = 0
    source_type: str = "seed"
    priority: int = 0
    parent_url: str | None = None
    context: CrawlTaskContext | None = None

    def __post_init__(self) -> None:
        """Normalize direct construction input at the crawl-task boundary."""

        normalized_source_name = str(self.source_name).strip().lower()
        if not normalized_source_name:
            raise ValueError("crawl task source_name is required")
        object.__setattr__(self, "source_name", normalized_source_name)

    def clone(
        self,
        *,
        url: str | None = None,
        source_name: str | None = None,
        kind: str | MediaKind | None = None,
        depth: int | None = None,
        source_type: str | None = None,
        priority: int | None = None,
        parent_url: str | None = None,
        task_id: str | None = None,
        context: CrawlTaskContext | Mapping[str, object] | None = None,
    ) -> "CrawlTask":
        """Return an immutable copy with selected fields replaced."""

        resolved_task_id = (
            self.task_id if task_id is None else _clean_optional_text(task_id)
        )
        return CrawlTask(
            url=self.url if url is None else url,
            source_name=(
                self.source_name if source_name is None else source_name
            ),
            task_id=resolved_task_id,
            kind=self.kind if kind is None else MediaKind.parse(kind),
            depth=self.depth if depth is None else depth,
            source_type=(
                self.source_type if source_type is None else source_type
            ),
            priority=(self.priority if priority is None else int(priority)),
            parent_url=(self.parent_url if parent_url is None else parent_url),
            context=(
                self.context
                if context is None
                else coerce_crawl_task_context(context)
            ),
        )

    def ensure_id(
        self,
        *,
        id_generator: IdGenerator,
    ) -> "CrawlTask":
        """Return this task with a stable non-empty identifier."""

        if _clean_optional_text(self.task_id) is not None:
            return self
        return self.clone(
            task_id=generate_task_id(id_generator=id_generator),
        )

    @classmethod
    def build(
        cls,
        *,
        request: "CrawlTaskBuildRequest",
        id_generator: "IdGenerator",
    ) -> "CrawlTask":
        """Build a normalized task from one explicit construction request."""

        depth = (
            request.depth
            if request.depth is not None
            else request.default_depth
        )
        priority = (
            request.priority
            if request.priority is not None
            else request.default_priority
        )
        task_id = _clean_optional_text(request.task_id) or generate_task_id(
            id_generator=id_generator
        )
        return cls(
            url=_clean_required_url(request.url),
            source_name=_clean_required_source_name(request.source_name),
            task_id=task_id,
            kind=MediaKind.parse(
                request.default_kind if request.kind is None else request.kind
            ),
            depth=max(0, int(depth)),
            source_type=(
                _clean_optional_text(request.source_type)
                or request.default_source_type
            ),
            priority=int(priority),
            parent_url=_clean_optional_text(request.parent_url),
            context=coerce_crawl_task_context(request.context),
        )

    @classmethod
    def build_seed(
        cls,
        *,
        url: str,
        source_name: str,
        source_type: str,
        kind: "str | MediaKind | None" = None,
        id_generator: "IdGenerator",
    ) -> "CrawlTask":
        return cls.build(
            request=CrawlTaskBuildRequest(
                url=url,
                source_name=source_name,
                kind=kind,
                source_type=source_type,
                default_source_type=source_type,
            ),
            id_generator=id_generator,
        )

    @classmethod
    def build_seeds(
        cls,
        *,
        seed_entries: "Iterable[object]",
        seed_source_type: str,
        id_generator: "IdGenerator",
    ) -> tuple["CrawlTask", ...]:
        tasks: list["CrawlTask"] = []
        for entry in seed_entries:
            if isinstance(entry, Mapping):
                raw_source_name = entry.get("source_name")
                raw_url = entry.get("url")
            else:
                raw_source_name = getattr(entry, "source_name", None)
                raw_url = getattr(entry, "url", None)

            source_name = _clean_required_source_name(raw_source_name)
            url = _clean_required_url(str(raw_url or ""))
            tasks.append(
                cls.build_seed(
                    url=url,
                    source_name=source_name,
                    source_type=seed_source_type,
                    kind=_guess_seed_kind(url),
                    id_generator=id_generator,
                )
            )
        return tuple(tasks)

    @classmethod
    def build_discovered(
        cls,
        *,
        source_name: str,
        url: str,
        kind: "str | MediaKind | None",
        parent_depth: int,
        source_type: str,
        parent_url: str,
        context: "CrawlTaskContext | Mapping[str, object] | None" = None,
        priority: "int | None" = None,
        default_kind: "str | MediaKind" = MediaKind.PAGE,
        id_generator: "IdGenerator",
    ) -> "CrawlTask":
        resolved_kind = MediaKind.parse(default_kind if kind is None else kind)
        resolved_depth = max(0, int(parent_depth))

        if resolved_kind in {MediaKind.PAGE, MediaKind.FEED}:
            resolved_depth += 1

        return cls.build(
            request=CrawlTaskBuildRequest(
                url=url,
                source_name=source_name,
                kind=resolved_kind,
                depth=resolved_depth,
                source_type=source_type,
                priority=priority,
                parent_url=parent_url,
                context=context,
                default_kind=resolved_kind,
                default_depth=resolved_depth,
                default_source_type=source_type,
            ),
            id_generator=id_generator,
        )

    @classmethod
    def from_mapping(
        cls,
        *,
        payload: "Mapping[str, object]",
        default_kind: "str | MediaKind" = MediaKind.PAGE,
        default_source_type: str = "seed",
        default_priority: int = 0,
        priority_resolver: "Callable[[CrawlTask], int] | None" = None,
        id_generator: "IdGenerator",
    ) -> "CrawlTask | None":
        url = _clean_optional_text(payload.get("url"))
        if url is None:
            return None

        source_name = _clean_optional_text(payload.get("source_name"))
        if source_name is None:
            raise ValueError("persisted crawl task is missing source_name")

        priority = _coerce_int(payload.get("priority"))
        raw_context = payload.get("context")
        context = (
            raw_context
            if raw_context is None
            or isinstance(raw_context, (CrawlTaskContext, Mapping))
            else None
        )

        task = cls.build(
            request=CrawlTaskBuildRequest(
                url=url,
                source_name=source_name,
                kind=_clean_optional_text(payload.get("kind")),
                depth=_coerce_int(payload.get("depth")),
                source_type=_clean_optional_text(payload.get("source_type")),
                priority=priority,
                parent_url=_clean_optional_text(payload.get("parent_url")),
                task_id=_clean_optional_text(payload.get("task_id")),
                context=context,
                default_kind=default_kind,
                default_source_type=default_source_type,
                default_priority=default_priority,
            ),
            id_generator=id_generator,
        )
        if priority is not None or priority_resolver is None:
            return task
        return task.clone(priority=int(priority_resolver(task)))

    @staticmethod
    def with_url_and_preserved_priority(
        *,
        task: "CrawlTask",
        url: str,
        priority: "int | None" = None,
    ) -> "CrawlTask":
        return task.clone(
            url=url,
            priority=task.priority if priority is None else int(priority),
        )

    @staticmethod
    def with_url_and_resolved_priority(
        *,
        task: "CrawlTask",
        url: str,
        priority_resolver: "Callable[[CrawlTask], int]",
    ) -> "CrawlTask":
        candidate = task.clone(url=url)
        return candidate.clone(
            priority=int(priority_resolver(candidate)),
        )

    @staticmethod
    def prepare_for_enqueue(
        *,
        task: "CrawlTask",
        priority_resolver: "Callable[[CrawlTask], int]",
    ) -> "CrawlTask":
        return task.clone(
            priority=int(priority_resolver(task)),
        )


@dataclass(frozen=True, slots=True)
class CrawlTaskBuildRequest:
    """Normalized input for constructing one crawl task."""

    url: str
    source_name: str
    kind: str | MediaKind | None = None
    depth: int | None = None
    source_type: str | None = None
    priority: int | None = None
    parent_url: str | None = None
    task_id: str | None = None
    context: "CrawlTaskContext | Mapping[str, object] | None" = None
    default_kind: str | MediaKind = MediaKind.PAGE
    default_depth: int = 0
    default_source_type: str = "seed"
    default_priority: int = 0


__all__ = [
    "CrawlTask",
    "CrawlTaskBuildRequest",
    "generate_task_id",
    "_clean_optional_text",
    "_clean_required_url",
    "_clean_required_source_name",
    "_coerce_int",
    "_strip_url_suffix",
    "_guess_seed_kind",
]
