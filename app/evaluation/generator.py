"""Deterministic ReportLab generator for the synthetic evaluation corpus."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from textwrap import wrap
from typing import Any

from app.evaluation.scenarios import Scenario, SyntheticDocument, get_scenarios


class EvaluationDataGenerationError(RuntimeError):
    """Raised when deterministic evaluation artifacts cannot be generated."""


def _write_pdf(path: Path, document: SyntheticDocument, bundle_id: str) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise EvaluationDataGenerationError(
            "ReportLab is required to generate evaluation PDFs"
        ) from exc

    width, height = A4
    pdf = canvas.Canvas(
        str(path),
        pagesize=A4,
        pageCompression=1,
        invariant=1,
    )
    pdf.setAuthor("EvidenceFlow")
    pdf.setCreator("EvidenceFlow deterministic evaluation generator")
    pdf.setTitle(document.title)
    pdf.setSubject(f"Synthetic evaluation document for {bundle_id}")

    y = height - 68
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(56, y, document.title)
    y -= 30
    pdf.setFont("Helvetica", 10)
    for source_line in document.lines:
        wrapped_lines = wrap(
            source_line,
            width=92,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        for line in wrapped_lines:
            if y < 72:
                pdf.setFont("Helvetica", 8)
                pdf.drawRightString(width - 56, 36, f"{bundle_id} | Page 1")
                pdf.showPage()
                y = height - 68
                pdf.setFont("Helvetica", 10)
            pdf.drawString(56, y, line)
            y -= 16
        y -= 3

    pdf.setFont("Helvetica", 8)
    pdf.setFillGray(0.35)
    pdf.drawString(56, 36, "SYNTHETIC DATA - NOT A REAL BUSINESS RECORD")
    pdf.drawRightString(width - 56, 36, f"{bundle_id} | Page 1")
    pdf.save()


def _write_ground_truth(path: Path, scenario: Scenario) -> None:
    payload: dict[str, Any] = scenario.as_ground_truth()
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def generate_evaluation_data(
    output_root: str | Path = "eval/bundles",
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Generate all 20 bundles and atomically replace ``output_root``.

    Generation happens in a sibling temporary directory, so a failed PDF write
    never leaves a half-generated corpus at the destination. The returned paths
    are ordered by bundle ID.
    """

    destination = Path(output_root).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"{destination} is not empty; pass overwrite=True to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    generated_names: list[str] = []
    try:
        for scenario in get_scenarios():
            bundle_path = temporary / scenario.bundle_id
            document_path = bundle_path / "documents"
            document_path.mkdir(parents=True)
            for document in scenario.documents:
                _write_pdf(
                    document_path / document.file_name,
                    document,
                    scenario.bundle_id,
                )
            _write_ground_truth(bundle_path / "ground_truth.json", scenario)
            generated_names.append(scenario.bundle_id)

        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return [destination / name for name in generated_names]
