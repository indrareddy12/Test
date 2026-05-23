"""
Greencare AI — Lego 1: Celery Worker
Orchestrates the document processing pipeline:
  1. Send file to Lego 2 (CPU triage / fast-track)
  2. If image/scanned, forward cleaned image to Lego 3 (Groq Vision)
  3. Write extracted JSON to ./pending_review/ for Lego 4 (HITL)
"""

import os
import json
import time
import logging

import requests
from celery import Celery

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "greencare_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# ---------------------------------------------------------------------------
# Service URLs (overridable via environment for Docker networking)
# ---------------------------------------------------------------------------
LEGO2_URL = os.environ.get("LEGO2_URL", "http://localhost:8001")
LEGO3_URL = os.environ.get("LEGO3_URL", "http://localhost:8002")

PENDING_DIR = "./pending_review"
os.makedirs(PENDING_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: retry-aware HTTP request
# ---------------------------------------------------------------------------
def _post_with_retry(url: str, retries: int = 3, **kwargs) -> dict:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, timeout=120, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)  # exponential back-off


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------
@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def process_document_task(self, file_path: str, job_id: str):
    """
    Pipeline orchestrator.
    Returns a dict describing the final outcome.
    """
    logger.info("[%s] Pipeline started. File: %s", job_id, file_path)

    try:
        # ------------------------------------------------------------------
        # Step 1: CPU Triage (Lego 2)
        # ------------------------------------------------------------------
        logger.info("[%s] → Lego 2 triage", job_id)
        with open(file_path, "rb") as fh:
            triage_resp = _post_with_retry(
                f"{LEGO2_URL}/triage",
                files={"file": (os.path.basename(file_path), fh)},
            )

        routing = triage_resp.get("routing")
        logger.info("[%s] Lego 2 routing decision: %s", job_id, routing)

        # ------------------------------------------------------------------
        # Step 2: Route decision
        # ------------------------------------------------------------------
        if routing == "fast_track_complete":
            # Digital PDF — text extracted locally, no Groq call needed
            final_data = triage_resp.get("extracted_data", {})
            final_data["_pipeline"] = "fast_track_cpu"
            logger.info("[%s] Fast-tracked (no Groq API call used)", job_id)

        elif routing == "forward_to_vlm":
            # Image or scanned PDF — send cleaned file to Groq
            cleaned_path = triage_resp.get("cleaned_file_path", file_path)
            logger.info("[%s] → Lego 3 Groq Vision. Image: %s", job_id, cleaned_path)

            ai_resp = _post_with_retry(
                f"{LEGO3_URL}/extract",
                json={"image_path": cleaned_path},
            )
            final_data = ai_resp.get("data", {})
            final_data["_pipeline"] = "groq_vision"
            logger.info("[%s] Groq extraction complete", job_id)

        else:
            raise ValueError(f"Unknown routing decision from Lego 2: {routing}")

        # ------------------------------------------------------------------
        # Step 3: Persist for HITL review (Lego 4)
        # ------------------------------------------------------------------
        final_data["_job_id"] = job_id
        output_path = os.path.join(PENDING_DIR, f"{job_id}.json")
        with open(output_path, "w", encoding="utf-8") as out_fh:
            json.dump(final_data, out_fh, indent=4, ensure_ascii=False)

        logger.info("[%s] ✓ Written to pending_review: %s", job_id, output_path)
        return {"status": "success", "job_id": job_id, "pipeline": final_data.get("_pipeline")}

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", job_id, exc, exc_info=True)
        # Celery retry on transient errors
        raise self.retry(exc=exc)
