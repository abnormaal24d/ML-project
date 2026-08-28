"""Rasterize document pages and extract provenance-bearing OCR text."""

from __future__ import annotations

from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

from config.environment.runtime_environment import (
    local_app_data,
    poppler_environment_paths,
)
from preprocessing.media.ocr.ocr_engine import merge_ocr_results
from preprocessing.media.ocr.ocr_result import (
    OpticalCharacterRecognitionResult,
)
from preprocessing.provenance import hash_file

if TYPE_CHECKING:
    from preprocessing.media.ocr.ocr_engine import OcrEngine


class DocumentOcrExtractor:
    """Extract provenance-bearing OCR when no native preview is available."""

    def __init__(
        self,
        *,
        first_page: int = 1,
        last_page: int = 2,
        poppler_path: str | Path | None = None,
        ocr_engine: OcrEngine | None = None,
    ) -> None:
        self._first_page = max(1, int(first_page))
        self._last_page = max(self._first_page, int(last_page))
        self._poppler_path = self._resolve_poppler_path(poppler_path)
        self._ocr_engine = ocr_engine

    def extract(
        self,
        *,
        path: Path,
    ) -> OpticalCharacterRecognitionResult | None:
        if self._ocr_engine is None:
            return None

        # pdf2image / poppler are required at startup when document OCR is on.
        from pdf2image import convert_from_path
        from pdf2image.exceptions import (
            PDFInfoNotInstalledError,
            PDFPageCountError,
            PDFSyntaxError,
        )

        try:
            source_hash = hash_file(path)
            if self._poppler_path is None:
                images = convert_from_path(
                    str(path),
                    first_page=self._first_page,
                    last_page=self._last_page,
                )
            else:
                images = convert_from_path(
                    str(path),
                    first_page=self._first_page,
                    last_page=self._last_page,
                    poppler_path=self._poppler_path,
                )
        except (
            PDFInfoNotInstalledError,
            PDFPageCountError,
            PDFSyntaxError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ):
            return None

        page_results: list[OpticalCharacterRecognitionResult] = []
        for image in images:
            result = self._ocr_engine.extract_pil(
                image=image,
                source_hash=source_hash,
            )
            if result is not None:
                page_results.append(result)
        if not page_results:
            return None
        if len(page_results) == 1:
            return page_results[0]

        return merge_ocr_results(results=tuple(page_results))

    @staticmethod
    def _resolve_poppler_path(
        configured_path: str | Path | None,
    ) -> str | None:
        candidate_paths: list[Path] = []
        if configured_path is not None:
            candidate_paths.append(Path(configured_path))
        candidate_paths.extend(poppler_environment_paths())
        if pdfinfo_from_path := which("pdfinfo"):
            candidate_paths.append(Path(pdfinfo_from_path).parent)

        app_data = local_app_data()
        if app_data is not None:
            winget_packages = app_data / "Microsoft" / "WinGet" / "Packages"
            if winget_packages.exists():
                candidate_paths.extend(
                    path.parent
                    for path in winget_packages.rglob("pdfinfo.exe")
                )
        candidate_paths.extend(
            (
                Path("C:/Program Files/poppler/Library/bin"),
                Path("C:/Program Files/Poppler/Library/bin"),
                Path("C:/poppler/Library/bin"),
                Path("C:/ProgramData/chocolatey/bin"),
            )
        )

        seen: set[str] = set()
        for candidate in candidate_paths:
            normalized = (
                str(candidate.resolve())
                if candidate.exists()
                else str(candidate)
            )
            if normalized in seen:
                continue
            seen.add(normalized)
            if (candidate / "pdfinfo.exe").exists() and (
                candidate / "pdftoppm.exe"
            ).exists():
                return str(candidate)
        return None
