"""
Greencare AI — Lego 1: API Gateway + Data Serialization (The Baseplate)
=======================================================================
Responsibilities (per blueprint §3.1):
  • Primary public API ingestion gateway
  • Token-based authorization
  • Directs execution queues (via FastAPI BackgroundTasks, no Redis needed)
  • Multi-page table stitching heuristic (§5, Algorithm)
  • Compiles raw AI outputs into hierarchical JSON, CSV, and Excel tables
  • Job status tracking
  • Image asset cropping from visual grounding coordinates (§4.3)

Runs on: Port 8000
"""

import os
import uuid
import shutil
import json
import logging
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import cv2
import numpy as np

from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    BackgroundTasks, Depends, status, Query
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Optional JWT auth (graceful fallback if jose not installed) ────────────
try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SECRET_KEY       = os.environ.get("SECRET_KEY", "greencare-dev-secret-change-in-production")
ALGORITHM        = "HS256"
ACCESS_TOKEN_TTL = int(os.environ.get("TOKEN_TTL_MINUTES", "1440"))  # 24h default

LEGO2_URL = os.environ.get("LEGO2_URL", "http://localhost:8001")
LEGO3_URL = os.environ.get("LEGO3_URL", "http://localhost:8002")

UPLOAD_DIR   = "./temp_uploads"
PENDING_DIR  = "./pending_review"
FINAL_DIR    = "./final_database"
REJECTED_DIR = "./rejected"
ASSETS_DIR   = "./extracted_assets"   # cropped logo/image assets from visual grounding

for d in [UPLOAD_DIR, PENDING_DIR, FINAL_DIR, REJECTED_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

ALLOWED_EXT = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# In-memory job registry
job_registry: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — API Gateway (Lego 1)",
    description=(
        "Core baseplate gateway. Handles ingestion, authorization, "
        "multi-page table stitching, and JSON/CSV/Excel serialization."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Token auth helpers
# ---------------------------------------------------------------------------
def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_TTL)
    if JWT_AVAILABLE:
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    # Fallback: simple base64-ish token
    import base64
    return base64.b64encode(json.dumps(payload).encode()).decode()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """Validate Bearer JWT. Returns payload dict. Raises 401 on failure."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    token = credentials.credentials
    if JWT_AVAILABLE:
        try:
            return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    # Fallback: accept any token in dev mode
    return {"sub": "dev_user"}


# ---------------------------------------------------------------------------
# §5 Multi-Page Table Stitching Heuristic
# ---------------------------------------------------------------------------
class PageData:
    """Wrapper that mirrors the blueprint's PageData interface."""
    def __init__(self, page_dict: dict):
        self.raw = page_dict
        tables = page_dict.get("tables", [])
        self._tables = tables
        self.table_headers = tables[0]["headers"] if tables else []
        self.table_rows    = tables[0]["rows"]    if tables else []

    def has_open_table_node(self) -> bool:
        """True if page ends mid-table (last char is not a closing marker)."""
        return bool(self._tables)

    def starts_with_table(self) -> bool:
        return bool(self._tables)

    def discard_table_headers(self):
        self.table_rows = self._tables[0]["rows"] if self._tables else []
        self._tables[0]["headers"] = []


def stitch_multipage_tables(page_data_list: list[dict]) -> list[dict]:
    """
    Blueprint §5 — Multi-page table stitching heuristic.
    If adjacent pages share matching column headers and an open table node,
    merge row matrices and discard duplicate headers.
    """
    if len(page_data_list) < 2:
        return page_data_list

    wrapped = [PageData(p) for p in page_data_list]

    for i in range(len(wrapped) - 1):
        curr = wrapped[i]
        nxt  = wrapped[i + 1]

        if (curr.has_open_table_node()
                and nxt.starts_with_table()
                and curr.table_headers
                and curr.table_headers == nxt.table_headers):

            logger.info("Stitching table from page %d into page %d", i, i + 1)
            curr.table_rows.extend(nxt.table_rows)
            nxt.discard_table_headers()

            # Merge back into raw dict
            if wrapped[i].raw.get("tables"):
                wrapped[i].raw["tables"][0]["rows"] = curr.table_rows
            if wrapped[i + 1].raw.get("tables"):
                wrapped[i + 1].raw["tables"][0]["headers"] = []

    return [w.raw for w in wrapped]


# ---------------------------------------------------------------------------
# §4.3 Visual Grounding — Crop image assets from bounding boxes
# ---------------------------------------------------------------------------
def crop_visual_assets(source_image_path: str, grounding_results: list, job_id: str) -> list[str]:
    """
    Blueprint §4.3 — crop detected logos / diagrams from the source image
    using coordinates returned by the VLM.
    Returns list of saved asset paths.
    """
    if not source_image_path or not os.path.exists(source_image_path):
        return []

    image = cv2.imread(source_image_path)
    if image is None:
        return []

    saved_paths = []
    for idx, item in enumerate(grounding_results):
        box   = item.get("box_2d", [])
        label = item.get("label", f"asset_{idx}")
        if len(box) != 4:
            continue
        x_min, y_min, x_max, y_max = [int(c) for c in box]
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        asset_filename = f"{job_id}_asset_{idx}_{label}.jpg"
        asset_path     = os.path.join(ASSETS_DIR, asset_filename)
        cv2.imwrite(asset_path, crop)
        saved_paths.append(asset_path)
        logger.info("Cropped visual asset: %s", asset_path)

    return saved_paths


# ---------------------------------------------------------------------------
# Serialization helpers — JSON / CSV / Excel
# ---------------------------------------------------------------------------
def build_dataframes(extracted_data: dict) -> dict[str, pd.DataFrame]:
    """Build a dict of {sheet_name: DataFrame} from the extracted JSON."""
    frames = {}

    # Key-value pairs → flat table
    kvp = extracted_data.get("key_value_pairs", {})
    if kvp:
        frames["Key_Value_Pairs"] = pd.DataFrame(
            [{"Field": k, "Value": v} for k, v in kvp.items()]
        )

    # Text block
    text = extracted_data.get("extracted_text", "")
    if text:
        frames["Extracted_Text"] = pd.DataFrame([{"Text": text}])

    # Tables
    for idx, table in enumerate(extracted_data.get("tables", [])):
        headers = table.get("headers", [])
        rows    = table.get("rows", [])
        if rows:
            frames[f"Table_{idx + 1}"] = pd.DataFrame(rows, columns=headers or None)

    return frames


def export_to_excel(extracted_data: dict) -> bytes:
    """Serialize extracted data to a multi-sheet Excel workbook."""
    frames = build_dataframes(extracted_data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if not frames:
            pd.DataFrame([{"result": "No structured data extracted"}]).to_excel(
                writer, sheet_name="Result", index=False
            )
        for sheet_name, df in frames.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()


def export_to_csv(extracted_data: dict) -> str:
    """Serialize all tables into a single concatenated CSV."""
    frames = build_dataframes(extracted_data)
    if not frames:
        return "No structured data extracted"
    sections = []
    for name, df in frames.items():
        sections.append(f"### {name}")
        sections.append(df.to_csv(index=False))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Core pipeline (runs in FastAPI BackgroundTask)
# ---------------------------------------------------------------------------
def run_pipeline(file_path: str, job_id: str, source_filename: str):
    logger.info("[%s] ▶ Pipeline started: %s", job_id, source_filename)
    job_registry[job_id] = {"status": "processing", "filename": source_filename}

    try:
        # ── Step 1: CPU Triage (Lego 2) ────────────────────────────────
        with open(file_path, "rb") as fh:
            triage_resp = requests.post(
                f"{LEGO2_URL}/triage",
                files={"file": (source_filename, fh)},
                timeout=90,
            )
        triage_resp.raise_for_status()
        triage = triage_resp.json()
        routing = triage.get("routing")
        logger.info("[%s] Routing: %s", job_id, routing)

        # ── Step 2: Route decision ─────────────────────────────────────
        if routing == "fast_track_complete":
            pages     = triage.get("pages", [triage.get("extracted_data", {})])
            pages     = stitch_multipage_tables(pages)
            final_data = {"pages": pages, "_pipeline": "fast_track_cpu"}

        elif routing == "forward_to_vlm":
            cleaned_path = triage.get("cleaned_file_path", file_path)
            ai_resp = requests.post(
                f"{LEGO3_URL}/extract",
                json={"image_path": cleaned_path},
                timeout=180,
            )
            ai_resp.raise_for_status()
            ai_data = ai_resp.json()
            page_data = ai_data.get("data", {})

            # Visual grounding → crop assets
            grounding = page_data.pop("visual_grounding", [])
            asset_paths = crop_visual_assets(cleaned_path, grounding, job_id)

            final_data = {
                "pages"        : [page_data],
                "_pipeline"    : "groq_vision",
                "_model_used"  : ai_data.get("model_used", ""),
                "_asset_paths" : asset_paths,
            }

        else:
            raise ValueError(f"Unknown routing: {routing}")

        # ── Step 3: Persist enriched JSON ─────────────────────────────
        final_data["_job_id"]    = job_id
        final_data["_filename"]  = source_filename
        final_data["_timestamp"] = datetime.utcnow().isoformat() + "Z"

        out_path = os.path.join(PENDING_DIR, f"{job_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        job_registry[job_id] = {
            "status"   : "pending_review",
            "pipeline" : final_data.get("_pipeline"),
            "filename" : source_filename,
        }
        logger.info("[%s] ✅ Written to pending_review", job_id)

    except Exception as exc:
        logger.error("[%s] ❌ Pipeline error: %s", job_id, exc, exc_info=True)
        job_registry[job_id] = {
            "status"   : "failed",
            "error"    : str(exc),
            "filename" : source_filename,
        }


# ---------------------------------------------------------------------------
# AUTH ENDPOINTS
# ---------------------------------------------------------------------------
class TokenRequest(BaseModel):
    client_id: str
    client_secret: str

VALID_CLIENTS = {
    os.environ.get("CLIENT_ID", "greencare_client"):
    os.environ.get("CLIENT_SECRET", "greencare_secret_dev"),
}

@app.post("/auth/token", tags=["Authentication"])
def get_token(req: TokenRequest):
    """Issue a Bearer JWT for API access."""
    if VALID_CLIENTS.get(req.client_id) != req.client_secret:
        raise HTTPException(status_code=401, detail="Invalid client credentials")
    token = create_access_token({"sub": req.client_id})
    return {"access_token": token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_TTL * 60}


# ---------------------------------------------------------------------------
# INGEST
# ---------------------------------------------------------------------------
@app.post("/api/v1/ingest", tags=["Ingestion"], status_code=202)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # Token auth — comment out Depends to run unauthenticated in dev
    # _user: dict = Depends(verify_token),
):
    """Accept a document upload and start the processing pipeline."""
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext}'")

    job_id    = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    background_tasks.add_task(run_pipeline, file_path, job_id, file.filename)

    return JSONResponse(status_code=202, content={
        "status"  : "queued",
        "job_id"  : job_id,
        "message" : "Processing started. Poll /api/v1/status/{job_id} for updates.",
    })


# ---------------------------------------------------------------------------
# STATUS & EXPORT
# ---------------------------------------------------------------------------
@app.get("/api/v1/status/{job_id}", tags=["Monitoring"])
def get_status(job_id: str):
    if job_id in job_registry:
        return {"job_id": job_id, **job_registry[job_id]}
    if os.path.exists(os.path.join(FINAL_DIR, f"{job_id}.json")):
        return {"job_id": job_id, "status": "committed"}
    if os.path.exists(os.path.join(PENDING_DIR, f"{job_id}.json")):
        return {"job_id": job_id, "status": "pending_review"}
    return {"job_id": job_id, "status": "not_found"}


@app.get("/api/v1/jobs", tags=["Monitoring"])
def list_jobs():
    return {"total": len(job_registry), "jobs": job_registry}


def _load_job_data(job_id: str) -> dict:
    """Load finalized or pending JSON for a job."""
    for folder in [FINAL_DIR, PENDING_DIR]:
        path = os.path.join(folder, f"{job_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found or still processing")


@app.get("/api/v1/export/{job_id}/json", tags=["Export"])
def export_json(job_id: str):
    """Download extracted data as JSON."""
    data = _load_job_data(job_id)
    return JSONResponse(content=data)


@app.get("/api/v1/export/{job_id}/csv", tags=["Export"])
def export_csv(job_id: str):
    """Download extracted data as CSV."""
    data = _load_job_data(job_id)
    pages = data.get("pages", [data])
    merged: dict = {}
    for p in pages:
        merged.update(p)
    csv_content = export_to_csv(merged)
    return StreamingResponse(
        io.BytesIO(csv_content.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={job_id}.csv"},
    )


@app.get("/api/v1/export/{job_id}/excel", tags=["Export"])
def export_excel(job_id: str):
    """Download extracted data as Excel workbook."""
    data = _load_job_data(job_id)
    pages = data.get("pages", [data])
    merged: dict = {}
    for p in pages:
        merged.update(p)
    xlsx_bytes = export_to_excel(merged)
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={job_id}.xlsx"},
    )


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status" : "ok",
        "service": "lego1-gateway",
        "version": "2.0.0",
        "queued" : sum(1 for v in job_registry.values() if v.get("status") == "processing"),
    }
