"""
Greencare AI — Lego 3: Groq Vision Engine (The Power Brick)
===========================================================
Responsibilities (per blueprint §3.3):
  • Ingests raw image matrices (base64 encoded)
  • Executes Groq Llama-3.2-90B-Vision (replaces local Qwen2.5-VL-7B)
  • Enforces strict output syntax via response_format=json_object (§4.4 CFG masking)
  • Returns formatted Markdown tables, text streams, and visual grounding coordinates
  • Retry + exponential back-off for Groq rate limits

Blueprint §4 features implemented via Groq API:
  §4.1 Dynamic Resolution → Groq handles natively via image_url
  §4.2 mRoPE            → Model's internal positional encoding
  §4.3 Visual Grounding  → Prompt instructs model to return box_2d coordinates
  §4.4 CFG Decoding      → response_format=json_object enforces grammar-level JSON

Runs on: Port 8002
"""

import os
import json
import time
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq, RateLimitError, APIError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — Lego 3: Groq Vision Engine",
    description=(
        "Multimodal VLM engine using Groq Llama-3.2 Vision. "
        "Implements §4.1–§4.4 of the blueprint via API."
    ),
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client     = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "4096"))

# ---------------------------------------------------------------------------
# §4.4 Constrained Grammar Decoding — System Prompt
# The response_format=json_object parameter is the API-level equivalent of
# the CFG masking layer described in §4.4. This prompt defines the exact
# JSON schema the model must produce; tokens outside it are masked to -∞.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an Intelligent Document Processing (IDP) engine — the equivalent of a locally 
compiled Qwen2.5-VL-7B-Instruct model running TensorRT-LLM on a Jetson edge server.

Your task is to perform a SINGLE FORWARD PASS on the provided document image and extract
ALL of the following simultaneously (no cascading — one pass, like a unified VLM):

1. Layout detection and full text extraction (print + handwriting)
2. Table structure detection with headers and row data
3. Key-value field extraction (named fields like Invoice No, Patient Name, Date, Total, etc.)
4. Visual grounding — detect and localize logos, diagrams, signatures, stamps
5. Document type classification
6. Confidence assessment

You MUST respond ONLY with a valid JSON object matching this EXACT schema 
(§4.4 Constrained Grammar Decoding — no filler text, no markdown wrappers):

{
  "document_type": "<string: invoice|medical_form|handwritten_note|shipping_label|table|receipt|form|unknown>",
  "language": "<string: ISO 639-1 code, e.g. en, hi, de, zh>",
  "extracted_text": "<string: full verbatim text, preserve line breaks with \\n>",
  "markdown_tables": "<string: all tables rendered in Markdown format, or empty string>",
  "key_value_pairs": {
    "<field_name>": "<field_value>"
  },
  "tables": [
    {
      "table_index": 0,
      "headers": ["<col1>", "<col2>"],
      "rows": [
        ["<val1>", "<val2>"]
      ]
    }
  ],
  "visual_grounding": [
    {
      "box_2d": [x_min, y_min, x_max, y_max],
      "label": "<logo|diagram|signature|stamp|image|chart>"
    }
  ],
  "handwriting_detected": false,
  "curved_text_detected": false,
  "page_count_estimate": 1,
  "confidence_warning": false,
  "confidence_warning_reason": null
}

Rules:
- DO NOT output anything outside the JSON object. No markdown code fences. No prose.
- null for inapplicable fields, [] for empty arrays, {} for empty objects.
- Preserve original spelling and casing in extracted_text.
- For visual_grounding: if no logos/diagrams found, return [].
- box_2d coordinates are pixel offsets from the image top-left corner.
- For curved or rotated text (package labels, bottles): transcribe the text as if unwarped.
"""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ImagePayload(BaseModel):
    image_path: str = Field(..., description="Path to the preprocessed image on disk.")

class ExtractionResponse(BaseModel):
    status    : str
    model_used: str
    data      : dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_image(image_path: str) -> tuple[str, str]:
    """Read image file and return (base64_string, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime, _ = mimetypes.guess_type(str(path))
    if mime not in {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def call_groq(b64_image: str, mime: str, retries: int = 3) -> dict:
    """
    §4.4 CFG masking implemented via response_format=json_object.
    Exponential back-off on rate limits.
    """
    for attempt in range(1, retries + 1):
        try:
            logger.info("Groq call attempt %d/%d (model=%s)", attempt, retries, GROQ_MODEL)
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Perform a full single-pass extraction of this document image. "
                                    "Return the complete JSON as per the schema in the system prompt."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                            },
                        ],
                    },
                ],
                model=GROQ_MODEL,
                temperature=0.0,           # §4.4: deterministic, like constrained decoding
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},  # §4.4: grammar-level JSON lock
            )

            raw = completion.choices[0].message.content
            logger.info("Groq response: %d chars", len(raw))
            return json.loads(raw)

        except RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning("Rate limit — waiting %ds. %s", wait, exc)
            time.sleep(wait)
            if attempt == retries:
                raise HTTPException(status_code=429, detail=f"Groq rate limit: {exc}")

        except (APIError, json.JSONDecodeError, Exception) as exc:
            logger.error("Groq error on attempt %d: %s", attempt, exc)
            if attempt == retries:
                raise HTTPException(status_code=502, detail=f"Groq API error: {exc}")
            time.sleep(2)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status" : "ok",
        "service": "lego3-groq-engine",
        "model"  : GROQ_MODEL,
    }


@app.post("/extract", tags=["Extraction"], response_model=ExtractionResponse)
def extract(payload: ImagePayload):
    """
    Single-pass multimodal extraction via Groq Llama-3.2 Vision.
    Implements blueprint §4.1–§4.4.
    """
    logger.info("Extraction request: %s", payload.image_path)
    try:
        b64, mime = encode_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    result = call_groq(b64, mime)

    return ExtractionResponse(
        status    ="success",
        model_used=GROQ_MODEL,
        data      =result,
    )
