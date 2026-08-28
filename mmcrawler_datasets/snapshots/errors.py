"""Errors raised while constructing or publishing training snapshots."""


class SnapshotBuildError(RuntimeError):
    """A complete training snapshot could not be safely produced."""


__all__ = ["SnapshotBuildError"]
