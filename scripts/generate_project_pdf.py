"""Generate Hazard Waste Detection project guide PDF from repo metadata."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = _ROOT / "docs" / "Hazard_Waste_Detection_Project_Guide.pdf"

BRAND = colors.HexColor("#1a365d")
BRAND_MID = colors.HexColor("#2c5282")
BRAND_LIGHT = colors.HexColor("#ebf4ff")
MUTED = colors.HexColor("#4a5568")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontSize=26,
            leading=30,
            alignment=TA_CENTER,
            textColor=BRAND,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontSize=11,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=8,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=15,
            leading=19,
            spaceBefore=16,
            spaceAfter=8,
            textColor=BRAND,
            borderPadding=(0, 0, 4, 0),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=5,
            textColor=BRAND_MID,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            leftIndent=14,
            spaceAfter=3,
        ),
        "mono": ParagraphStyle(
            "Mono",
            parent=base["Code"],
            fontSize=7.5,
            leading=9.5,
            fontName="Courier",
            backColor=BRAND_LIGHT,
        ),
        "toc_num": ParagraphStyle(
            "TOCNum",
            parent=base["Normal"],
            fontSize=10,
            leading=18,
            textColor=BRAND_MID,
            fontName="Helvetica-Bold",
        ),
        "toc_item": ParagraphStyle(
            "TOCItem",
            parent=base["Normal"],
            fontSize=10,
            leading=18,
            leftIndent=28,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def _table(data: list[list[str]], col_widths: list[float] | None = None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_MID),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def _diagram(text: str) -> Preformatted:
    return Preformatted(text.strip() + "\n", _styles()["mono"])


def _section_bar(title: str, s: dict) -> list:
    """Coloured section header band."""
    bar = Table([[Paragraph(title, s["h1"])]], colWidths=[17 * cm])
    bar.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [bar, Spacer(1, 6)]


class NumberedCanvas:
    """Footer with page number."""

    def __init__(self, doc_title: str):
        self.doc_title = doc_title

    def draw(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, self.doc_title)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()


def build_pdf(path: Path) -> None:
    s = _styles()
    story: list = []
    today = datetime.now().strftime("%B %d, %Y")
    doc_title = "Hazardous Waste Detection — Project Guide"

    # ── Cover ──────────────────────────────────────────────────────────
    cover_band = Table(
        [[Paragraph("HAZARDOUS WASTE DETECTION", ParagraphStyle(
            "CoverBand", fontSize=11, textColor=colors.white,
            alignment=TA_CENTER, fontName="Helvetica-Bold",
        ))]],
        colWidths=[17 * cm],
    )
    cover_band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(Spacer(1, 2.5 * cm))
    story.append(cover_band)
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Project Guide &amp; Technical Reference", s["title"]))
    story.append(Paragraph(
        "Instance Segmentation for Cylinders &amp; Shock Absorbers<br/>in Scrap-Yard Imagery",
        s["subtitle"],
    ))
    story.append(Spacer(1, 0.8 * cm))
    story.append(
        _table(
            [
                ["Item", "Detail"],
                ["Production model", "YOLOv5s-Seg (Run 4)"],
                ["Comparison models", "YOLO11s (Run 2), YOLOv8s (Run 3)"],
                ["Dataset", "393 unique images — 275 / 79 / 39 split"],
                ["Live demo", "hazard-waste-detection.onrender.com"],
                ["Repository", "github.com/aravindhangen/hazard-waste-detection"],
            ],
            col_widths=[4.5 * cm, 12.5 * cm],
        )
    )
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph(f"Document version: 1.0  |  Generated: {today}", s["cover_meta"]))
    story.append(Paragraph("For academic review, onboarding, and technical understanding.", s["cover_meta"]))
    story.append(PageBreak())

    # ── Table of contents ──────────────────────────────────────────────
    story.append(Paragraph("On This Page", s["h1"]))
    story.append(Spacer(1, 6))
    toc_items = [
        ("01", "The Problem"),
        ("02", "A Solution"),
        ("03", "Where Else It Applies"),
        ("04", "Architecture (3 Views)"),
        ("05", "Project Pipeline"),
        ("06", "CLI Commands"),
        ("07", "Tech Stack"),
        ("08", "Libraries"),
        ("09", "Project Structure"),
        ("10", "File Reference"),
        ("11", "Key Takeaways"),
    ]
    for num, label in toc_items:
        row = Table(
            [[Paragraph(num, s["toc_num"]), Paragraph(label, s["toc_item"])]],
            colWidths=[1.2 * cm, 15.8 * cm],
        )
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(row)
    story.append(PageBreak())

    # ── 01 The Problem ─────────────────────────────────────────────────
    story.extend(_section_bar("01. The Problem", s))
    story.append(Paragraph(
        "Scrap yards and metal-recycling facilities process large volumes of mixed waste every day. "
        "Among ordinary scrap, two object types pose serious safety risks:",
        s["body"],
    ))
    story.append(Paragraph(
        "<b>Pressurized gas cylinders</b> — can explode when crushed, sheared, or compacted, "
        "injuring workers and damaging equipment.",
        s["bullet"],
    ))
    story.append(Paragraph(
        "<b>Automotive shock absorbers</b> — may rupture and release toxic hydraulic fluid "
        "during shredding or compression.",
        s["bullet"],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Why manual inspection fails:</b>", s["h2"]))
    for item in [
        "Slow and inconsistent at conveyor-belt speeds",
        "Hazards are often rusted, buried, or partially occluded",
        "Unsafe for workers to inspect close-up in active yards",
        "No scalable audit trail for compliance reporting",
    ]:
        story.append(Paragraph(f"• {item}", s["bullet"]))
    story.append(Spacer(1, 10))

    # ── 02 A Solution ──────────────────────────────────────────────────
    story.extend(_section_bar("02. A Solution", s))
    story.append(Paragraph(
        "An end-to-end <b>computer-vision system</b> that detects, localizes, and <b>segments</b> "
        "hazardous objects using YOLO-based instance segmentation — served through a "
        "<b>FastAPI</b> REST API and an interactive web dashboard.",
        s["body"],
    ))
    story.append(Spacer(1, 6))
    story.append(
        _table(
            [
                ["Component", "Description"],
                ["Detection", "Two classes: Cylinder, Shock Absorber"],
                ["Segmentation", "Pixel-level masks (not boxes only) for cluttered piles"],
                ["Alerts", "EXPLOSIVE (Cylinder) / TOXIC (Shock Absorber)"],
                ["API", "POST /predict, POST /predict/compare, GET /health"],
                ["Dashboard", "Upload, webcam, live scan, 3-model side-by-side compare"],
                ["Deployment", "Local (Docker), cloud (Render.com)"],
            ],
            col_widths=[3.5 * cm, 13.5 * cm],
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph("High-level solution flow:", s["h2"]))
    story.append(_diagram(
        """
Roboflow Annotation  -->  Dataset QA & Dedup  -->  YOLO Training
        |                                              |
        v                                              v
   2 classes                                   275/79/39 split
        |                                              |
        +----------------------+-----------------------+
                               v
                    FastAPI + Dashboard + Cloud Deploy
                               |
                               v
                    Real-time hazard alerts for operators
"""
    ))
    story.append(PageBreak())

    # ── 03 Where Else It Applies ───────────────────────────────────────
    story.extend(_section_bar("03. Where Else It Applies", s))
    applications = [
        ("Scrap-metal yards", "Pre-shredder hazard screening on feed conveyors"),
        ("Auto salvage facilities", "Cylinder and shock-absorber removal before crushing"),
        ("Municipal recycling", "Mixed-waste sorting lines with camera monitoring"),
        ("Industrial demolition", "Salvage pile audits before transport"),
        ("Compliance auditing", "Photo evidence for hazardous-waste regulations"),
        ("Remote expert review", "Upload imagery for off-site safety assessment"),
        ("Edge deployment (future)", "ONNX / Jetson on-premise GPU inference"),
    ]
    story.append(
        _table(
            [["Domain", "Use case"]] + [[a, u] for a, u in applications],
            col_widths=[5 * cm, 12 * cm],
        )
    )
    story.append(Spacer(1, 10))

    # ── 04 Architecture (3 Views) ──────────────────────────────────────
    story.extend(_section_bar("04. Architecture (3 Views)", s))

    story.append(Paragraph("View 1 — End-to-end ML pipeline", s["h2"]))
    story.append(_diagram(
        """
DATA ACQUISITION (scrap-yard photos)
        |
        v
ANNOTATION — Roboflow, 2 classes, YOLO polygon format
        |
        v
PIPELINE — merge, dedup (imagehash), validate, stratified split
        |          hazard_dataset/  -->  hazard_dataset_clean/
        v
TRAINING — YOLOv5s / YOLO11s / YOLOv8s segmentation (640px, 100 epochs)
        |
        v
EVALUATION — precision, recall, mAP@50, per-class metrics
        |
        v
DEPLOYMENT — FastAPI + dashboard + Render / Docker
"""
    ))

    story.append(Paragraph("View 2 — Runtime inference (deployment)", s["h2"]))
    story.append(_diagram(
        """
Browser (upload / webcam / compare mode)
        |
        v
dashboard/  —  app.js, index.html, styles.css
        |
        v  POST /predict  |  POST /predict/compare
api/main.py  —  FastAPI, CORS, static files, /docs
        |
        v
model_manager.py  —  lazy/eager load, LRU cache, routing
        |
   +----+---------+---------+
   v              v         v
yolov5_engine  ultralytics  ultralytics
 (Run 4)       YOLO11s      YOLOv8s
        |
        v
Masks + boxes + confidence + hazard_type JSON + annotated JPEG
"""
    ))

    story.append(Paragraph("View 3 — Module responsibilities", s["h2"]))
    story.append(
        _table(
            [
                ["Module", "Path", "Responsibility"],
                ["Config", "hazard_detection/config/", "Paths, classes, model catalog, API settings"],
                ["Label utils", "hazard_detection/data/", "Polygon validation, cleaning, counts"],
                ["Pipeline", "scripts/pipeline/00-07", "Build, audit, dedup, visualize, validate"],
                ["Training", "scripts/training/08-13", "Train runs, evaluate, compare models"],
                ["API", "api/main.py", "HTTP routes, schemas, static dashboard mount"],
                ["Inference", "api/inference.py", "Image decode, mask overlay, hazard logic"],
                ["YOLOv5 engine", "api/yolov5_engine.py", "Production Run 4 weights"],
                ["Ultralytics", "api/ultralytics_engine.py", "YOLO11s / YOLOv8s inference"],
                ["Dashboard", "dashboard/", "Single-model + 3-model compare UI"],
                ["Deploy", "Dockerfile*, render.yaml", "Local GPU/CPU and Render cloud"],
            ],
            col_widths=[2.2 * cm, 5.3 * cm, 9.5 * cm],
        )
    )
    story.append(PageBreak())

    # ── 05 Project Pipeline ────────────────────────────────────────────
    story.extend(_section_bar("05. Project Pipeline", s))
    story.append(
        _table(
            [
                ["Phase", "Academic task", "Key scripts / artifacts"],
                ["1", "Planning", "docs/PROJECT_PROPOSAL.md, PROJECT_ARCHITECTURE.md"],
                ["2", "Annotation", "Roboflow -> 00_build_dataset.py -> hazard_dataset/"],
                ["3", "Augment & balance", "05_deduplicate_and_resplit.py -> hazard_dataset_clean/"],
                ["4", "Model selection", "Benchmark YOLOv5s / YOLO11s / YOLOv8s"],
                ["5", "Train & evaluate", "13_train_yolov5_run4.py, 09/11, 12_compare"],
                ["6", "Deploy & test", "api/main.py, verify_endpoints.py, Render"],
            ],
            col_widths=[1.4 * cm, 3.2 * cm, 12.4 * cm],
        )
    )
    story.append(Spacer(1, 10))
    story.append(Paragraph("Frozen dataset statistics", s["h2"]))
    story.append(
        _table(
            [
                ["Metric", "Value"],
                ["Unique images (after dedup)", "393"],
                ["Train / Val / Test", "275 / 79 / 39 (70 / 20 / 10 %)"],
                ["Classes", "Cylinder (0), Shock_Absorber (1)"],
                ["Instance ratio", "1,167 : 997 (~1.17:1 — balanced)"],
                ["Cross-split leakage", "0 duplicates"],
                ["Malformed labels (final)", "0"],
            ],
            col_widths=[6 * cm, 11 * cm],
        )
    )
    story.append(Spacer(1, 10))
    story.append(Paragraph("Model performance (validation set — target 75-80% precision & recall)", s["h2"]))
    story.append(
        _table(
            [
                ["Model", "Precision", "Recall", "Cylinder Recall", "Role"],
                ["YOLOv5s-Seg", "76.0%", "75.0%", "75.0%", "Production"],
                ["YOLO11s-Seg", "77.0%", "80.0%", "80.0%", "Benchmark"],
                ["YOLOv8s-Seg", "78.0%", "77.0%", "78.0%", "Benchmark"],
            ],
            col_widths=[3.5 * cm, 2.5 * cm, 2.5 * cm, 3.5 * cm, 5 * cm],
        )
    )
    story.append(PageBreak())

    # ── 06 CLI Commands ────────────────────────────────────────────────
    story.extend(_section_bar("06. CLI Commands", s))
    story.append(Paragraph("Run all commands from the project root.", s["body"]))
    commands = [
        ("First-time setup", "python -m venv .venv"),
        ("Install (API + compare)", 'pip install -e ".[api,run2]"'),
        ("Install (full stack)", 'pip install -e ".[train,api,run2]"'),
        ("Start server (Windows)", "run_server.bat"),
        ("Start server (PowerShell)", ".\\run_server.ps1"),
        ("Start server (manual)", "uvicorn api.main:app --host 0.0.0.0 --port 8000"),
        ("API smoke test", "python scripts/verify_endpoints.py"),
        ("Merge Roboflow exports", "python scripts/pipeline/00_build_dataset.py"),
        ("Dedup & clean split", "python scripts/pipeline/05_deduplicate_and_resplit.py"),
        ("Validate clean dataset", "python scripts/pipeline/06_final_dataset_validation.py"),
        ("Train YOLOv5 Run 4", "python scripts/training/13_train_yolov5_run4.py --device 0"),
        ("Train YOLO11 Run 2", "python scripts/training/09_train_yolo11_run2.py --device 0"),
        ("Train YOLOv8 Run 3", "python scripts/training/11_train_yolov8_run3.py --device 0"),
        ("Compare all 3 models", "python scripts/training/12_compare_all_runs.py"),
        ("Regenerate this PDF", "python scripts/generate_project_pdf.py"),
        ("Docker (CPU)", "docker compose up --build"),
        ("Docker (GPU)", "docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build"),
    ]
    story.append(
        _table(
            [["Action", "Command"]] + [[a, c] for a, c in commands],
            col_widths=[4 * cm, 13 * cm],
        )
    )
    story.append(Spacer(1, 10))
    story.append(Paragraph("API endpoints (local: http://127.0.0.1:8000)", s["h2"]))
    story.append(
        _table(
            [
                ["Method", "Path", "Purpose"],
                ["GET", "/health", "Service health + model load status"],
                ["GET", "/models", "Model catalog and benchmark metrics"],
                ["POST", "/predict", "Single-image inference"],
                ["POST", "/predict/compare", "Multi-model side-by-side inference"],
                ["GET", "/docs", "Swagger UI (interactive API docs)"],
                ["GET", "/dashboard/", "Web UI (upload, webcam, compare)"],
            ],
            col_widths=[2 * cm, 5 * cm, 10 * cm],
        )
    )
    story.append(PageBreak())

    # ── 07 Tech Stack ────────────────────────────────────────────────────
    story.extend(_section_bar("07. Tech Stack", s))
    story.append(
        _table(
            [
                ["Layer", "Technology", "Notes"],
                ["Language", "Python 3.10+", "Core runtime"],
                ["Deep learning", "PyTorch", "Training & GPU inference"],
                ["Segmentation", "YOLOv5 / YOLO11 / YOLOv8", "Instance masks"],
                ["Annotation", "Roboflow", "Polygon labels, YOLO export"],
                ["Computer vision", "OpenCV, Pillow", "I/O, overlay, encoding"],
                ["Deduplication", "imagehash", "Perceptual-hash near-duplicate removal"],
                ["API framework", "FastAPI + Uvicorn", "Async REST server"],
                ["Validation", "Pydantic", "Request/response schemas"],
                ["Frontend", "HTML / CSS / JavaScript", "Dashboard (no framework)"],
                ["Containers", "Docker, Compose", "Local + GPU override"],
                ["Cloud", "Render.com", "render.yaml + Dockerfile.render"],
                ["Version control", "Git, GitHub", "Weights + reports in repo"],
                ["Testing", "httpx", "scripts/verify_endpoints.py"],
            ],
            col_widths=[3.2 * cm, 5.5 * cm, 8.3 * cm],
        )
    )
    story.append(Spacer(1, 10))
    story.append(Paragraph("Environment variables", s["h2"]))
    story.append(
        _table(
            [
                ["Variable", "Default", "Purpose"],
                ["HAZARD_DEFAULT_MODEL_ID", "yolov5", "Production model in API"],
                ["HAZARD_MODEL_WEIGHTS", "Run 4 best_yolov5s.pt", "Weight path override"],
                ["HAZARD_DATASET_ROOT", "hazard_dataset_clean/", "Training data root"],
                ["HAZARD_DEVICE", "0 / cpu", "CUDA device or CPU"],
                ["HAZARD_IMG_SIZE", "640 (416 on Render)", "Inference resolution"],
                ["PORT", "8000", "API listen port"],
            ],
            col_widths=[5 * cm, 5 * cm, 7 * cm],
        )
    )
    story.append(PageBreak())

    # ── 08 Libraries ─────────────────────────────────────────────────────
    story.extend(_section_bar("08. Libraries", s))
    story.append(Paragraph("Install profiles (pyproject.toml optional extras):", s["body"]))
    for line in [
        'pip install -e ".[api]"           — inference server only',
        'pip install -e ".[api,run2]"        — + Ultralytics (YOLO11 / YOLOv8)',
        'pip install -e ".[train,api,run2]"  — full training + API stack',
    ]:
        story.append(Paragraph(f"• {line}", s["bullet"]))
    story.append(Spacer(1, 8))
    story.append(
        _table(
            [
                ["Library", "Category", "Purpose in this project"],
                ["torch / torchvision", "ML", "Training, GPU inference, tensor ops"],
                ["ultralytics", "ML", "YOLO11s and YOLOv8s segmentation"],
                ["ultralytics/yolov5", "ML", "YOLOv5s-Seg production inference"],
                ["fastapi", "API", "REST endpoints, OpenAPI docs"],
                ["uvicorn", "API", "ASGI server"],
                ["python-multipart", "API", "File upload handling"],
                ["opencv-python", "CV", "Image decode, mask drawing"],
                ["Pillow", "CV", "JPEG encoding for API responses"],
                ["imagehash", "Data", "Perceptual-hash deduplication"],
                ["pydantic", "API", "Schema validation"],
                ["httpx", "Testing", "Async HTTP smoke tests"],
                ["albumentations", "Training", "Augmentation pipeline"],
                ["pycocotools", "Training", "COCO-style mAP evaluation"],
                ["pyyaml", "Config", "data.yaml dataset manifests"],
                ["reportlab", "Docs", "This PDF generator"],
            ],
            col_widths=[4 * cm, 2.5 * cm, 10.5 * cm],
        )
    )
    story.append(PageBreak())

    # ── 09 Project Structure ─────────────────────────────────────────────
    story.extend(_section_bar("09. Project Structure", s))
    story.append(_diagram(
        """
Hazard_Waste_Detection/
|
|-- api/                         FastAPI service (Task 6)
|   |-- main.py                  Routes: /health, /predict, /compare
|   |-- model_manager.py         Model load, cache, routing
|   |-- yolov5_engine.py         YOLOv5s Run 4 inference
|   |-- ultralytics_engine.py    YOLO11 / YOLOv8 inference
|   |-- inference.py             Image processing, hazard alerts
|   `-- schemas.py               Pydantic request/response models
|
|-- dashboard/                   Web UI (upload, webcam, compare)
|-- hazard_detection/            Core library
|   `-- config/                  paths.py, models.py, dataset.py
|
|-- scripts/
|   |-- pipeline/                00-07 dataset preparation
|   |-- training/                08-13 train & compare
|   |-- verify_endpoints.py      API smoke tests
|   `-- generate_project_pdf.py  This document generator
|
|-- hazard_dataset_clean/        Frozen dataset (data.yaml in Git)
|-- hazard_dataset/              Raw merged dataset (local only)
|
|-- runs/
|   |-- yolov5s_run4/            Production weights & evaluation
|   |-- yolo11s_run2/             YOLO11 benchmark
|   |-- yolov8s_run3/             YOLOv8 benchmark
|   `-- comparison/              3-model JSON / TXT reports
|
|-- docs/                        Reports, guides, this PDF
|-- evaluation_reports/          Legacy Run 1 evaluation
|-- Dockerfile / Dockerfile.render / render.yaml
`-- run_server.bat / .ps1        Windows launchers
"""
    ))
    story.append(PageBreak())

    # ── 10 File Reference ────────────────────────────────────────────────
    story.extend(_section_bar("10. File Reference", s))
    story.append(
        _table(
            [
                ["File / path", "Description"],
                ["api/main.py", "FastAPI application entry point"],
                ["api/model_manager.py", "Lazy/eager model loading and LRU cache"],
                ["api/yolov5_engine.py", "YOLOv5s-Seg Run 4 inference (torch.load patch)"],
                ["api/ultralytics_engine.py", "YOLO11s / YOLOv8s via Ultralytics API"],
                ["api/inference.py", "Decode uploads, resize, overlay masks, hazard JSON"],
                ["api/schemas.py", "Pydantic models for API I/O"],
                ["hazard_detection/config/models.py", "Model catalog and benchmark metrics"],
                ["hazard_detection/config/paths.py", "Canonical paths (weights, dataset, runs)"],
                ["hazard_detection/config/dataset.py", "CLASS_NAMES, Roboflow source metadata"],
                ["hazard_detection/data/labels.py", "YOLO polygon validation and cleaning"],
                ["scripts/pipeline/00_build_dataset.py", "Merge two Roboflow exports"],
                ["scripts/pipeline/05_deduplicate_and_resplit.py", "Dedup + stratified resplit"],
                ["scripts/pipeline/06_final_dataset_validation.py", "Final QA before training"],
                ["scripts/training/13_train_yolov5_run4.py", "Production YOLOv5s training"],
                ["scripts/training/12_compare_all_runs.py", "3-model validation comparison"],
                ["scripts/verify_endpoints.py", "Automated API endpoint tests"],
                ["runs/yolov5s_run4/weights/best_yolov5s.pt", "Production weights (~15 MB)"],
                ["runs/comparison/all_runs_comparison.json", "Benchmark summary (all models)"],
                ["hazard_dataset_clean/data.yaml", "YOLO training manifest (in Git)"],
                ["render.yaml", "Render.com Blueprint (env vars, Docker)"],
                ["Dockerfile.render", "CPU-optimised image for cloud deploy"],
                ["README.md", "Main documentation (Tasks 1-6, quick start)"],
                ["docs/PROJECT_PROPOSAL.md", "Academic proposal sections 1-11"],
                ["docs/TASK_DELIVERABLES.md", "Per-task deliverable checklist"],
                ["docs/Hazard_Waste_Detection_Project_Guide.pdf", "This document"],
            ],
            col_widths=[6.8 * cm, 10.2 * cm],
        )
    )
    story.append(PageBreak())

    # ── 11 Key Takeaways ─────────────────────────────────────────────────
    story.extend(_section_bar("11. Key Takeaways", s))
    takeaways = [
        ("Safety-first design",
         "Two hazard classes with distinct alert types: EXPLOSIVE (Cylinder) and TOXIC (Shock Absorber)."),
        ("Instance segmentation",
         "Pixel masks outperform bounding boxes in cluttered scrap-yard scenes."),
        ("Rigorous dataset engineering",
         "393 unique images, zero leakage, balanced classes — dedup via perceptual hashing."),
        ("Three-model benchmark",
         "YOLOv5s (production), YOLO11s, YOLOv8s evaluated on the same frozen split."),
        ("Performance target",
         "75-80% precision and recall on validation; Cylinder recall is the primary safety metric."),
        ("Production deployment",
         "FastAPI + dashboard deployed on Render; Docker for local/GPU environments."),
        ("Privacy by design",
         "Dataset images stay local-only; Git contains manifests, weights, and JSON reports."),
        ("Reproducible pipeline",
         "Numbered scripts (00-13) cover annotation through deployment end-to-end."),
        ("Extensible architecture",
         "ModelManager routes to YOLOv5 or Ultralytics engines without API contract changes."),
    ]
    story.append(
        _table(
            [["Topic", "Takeaway"]] + [[t, d] for t, d in takeaways],
            col_widths=[4 * cm, 13 * cm],
        )
    )
    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            f"<i>End of document — {doc_title}<br/>"
            "Regenerate: python scripts/generate_project_pdf.py</i>",
            s["footer"],
        )
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    numbered = NumberedCanvas(doc_title)

    def on_page(canvas, doc):
        numbered.draw(canvas, doc)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.2 * cm,
        title=doc_title,
        author="Hazard Waste Detection Project",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote: {path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT
    build_pdf(out)
