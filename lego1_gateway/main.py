"""
Greencare AI — Lego 1: API Gateway (Port 8000)
Receives uploaded documents, assigns a UUID job ID, and pushes the
processing task into the Celery / Redis queue.
"""

import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from worker import process_document_task

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — API Gateway",
    description="Accepts document uploads and routes them through the IDP pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "service": "lego1-gateway"}


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------
@app.post("/api/v1/ingest", tags=["Ingestion"], status_code=202)
async def ingest_document(file: UploadFile = File(...)):
    """
    Accept a document file (PDF or image), persist it to disk, and enqueue
    the processing task.  Returns immediately with a job_id for polling.
    """
    # Validate file extension
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    # Persist upload to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Enqueue background task
    task = process_document_task.delay(file_path, job_id)

    return JSONResponse(
        content={
            "status": "queued",
            "job_id": job_id,
            "celery_task_id": task.id,
            "message": "Document accepted. Check /api/v1/status/{job_id} for updates.",
        },
        status_code=202,
    )


# ---------------------------------------------------------------------------
# Status endpoint (reads from pending_review / final_database folders)
# ---------------------------------------------------------------------------
@app.get("/api/v1/status/{job_id}", tags=["Monitoring"])
async def get_job_status(job_id: str):
    """
    Simple file-based status check.
    - pending_review/{job_id}.json  → awaiting human approval
    - final_database/{job_id}.json  → approved and committed
    """
    pending_path = f"./pending_review/{job_id}.json"
    final_path = f"./final_database/{job_id}.json"

    if os.path.exists(final_path):
        return {"status": "committed", "job_id": job_id}
    elif os.path.exists(pending_path):
        return {"status": "pending_review", "job_id": job_id}
    else:
        return {"status": "processing", "job_id": job_id}
