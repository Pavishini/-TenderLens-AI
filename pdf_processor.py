"""PDF text extraction with page-level source tracking."""

from dataclasses import dataclass

import pymupdf as fitz


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be read or contains no extractable text."""


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def extract_pdf_pages(pdf_bytes: bytes) -> list[PageText]:
    """Extract selectable text from each page, retaining one-based page numbers.

    This intentionally does not perform OCR. Image-only/scanned PDFs therefore
    produce a clear error instead of appearing to have been analyzed.
    """
    if not pdf_bytes:
        raise PDFProcessingError("The uploaded PDF is empty.")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.is_encrypted:
                raise PDFProcessingError(
                    "This PDF is password-protected. Upload an unlocked PDF."
                )

            pages = [
                PageText(page_number=number, text=page.get_text("text").strip())
                for number, page in enumerate(document, start=1)
            ]
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            "The uploaded file could not be read as a PDF."
        ) from exc

    if not any(page.text for page in pages):
        raise PDFProcessingError(
            "No readable text was found. Scanned or image-only PDFs are not "
            "supported because OCR is not implemented."
        )

    return pages
