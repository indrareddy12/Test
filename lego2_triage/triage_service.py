"""
Greencare AI — Lego 2: Ingestion & Triage Vector Head (The Fast-Track)
======================================================================
Responsibilities (per blueprint §3.2):
  • Runs on cheap CPU — protects GPU servers from unnecessary calls
  • Inspects inbound files:
      - Digitally native PDF  → extract text + coordinate matrices via pypdf / pypdfium2
      - Scanned / photo       → apply glare suppression + deskew, forward to Groq (Lego 3)
  • Blueprint §5 Resiliency:
      - Specular Glare Suppression (adaptive threshold + TELEA inpaint + CLAHE)
      - Deskew correction (minAreaRect angle detection)

Runs on: Port 8001
"""

import io
import os
import shutil
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

# Optional: pypdfium2 for richer binary stream extraction
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False
    logging.warning("pypdfium2 not available — falling back to pypdf only")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — Lego 2: CPU Triage",
    description="Fast-track digital PDFs; preprocess images before forwarding to Groq.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TEMP_DIR = "./lego2_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

IMAGE_EXT       = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXT         = ".pdf"
MIN_TEXT_CHARS  = 50   # min chars to consider a PDF as "digital"


# ---------------------------------------------------------------------------
# §5 Specular Glare Suppression Filter
# ---------------------------------------------------------------------------
def apply_glare_suppression(image_path: str) -> str:
    """
    Blueprint §5 — Glare Suppression:
    Calculates an adaptive intensity threshold matrix.
    High-intensity glare regions are smoothed using neighboring pixel vectors
    (TELEA inpainting). Followed by CLAHE contrast enhancement.
    Returns path to cleaned image.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("Cannot read image for glare suppression: %s", image_path)
        return image_path

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive glare threshold (mean + 2.5σ of local region)
    mean_val = np.mean(gray)
    std_val  = np.std(gray)
    threshold = min(255, int(mean_val + 2.5 * std_val))
    threshold = max(threshold, 200)  # lower bound

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel  = np.ones((5, 5), np.uint8)
    mask    = cv2.dilate(mask, kernel, iterations=2)

    # TELEA inpainting to reconstruct glare regions from neighbors
    cleaned = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # CLAHE on L channel for contrast enhancement
    lab        = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l          = clahe.apply(l)
    cleaned    = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    base, ext     = os.path.splitext(image_path)
    clean_path    = f"{base}_glare_clean{ext}"
    cv2.imwrite(clean_path, cleaned)
    logger.info("Glare suppression complete → %s", clean_path)
    return clean_path


# ---------------------------------------------------------------------------
# Deskew correction
# ---------------------------------------------------------------------------
def deskew_image(image_path: str) -> str:
    """
    Blueprint §5 — Extreme Label Rotation Compensation pre-processing.
    Detects document skew angle via minAreaRect and corrects it.
    """
    image = cv2.imread(image_path)
    if image is None:
        return image_path

    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray   = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 10:
        return image_path

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    if abs(angle) < 0.5:
        return image_path

    h, w   = image.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

    base, ext   = os.path.splitext(image_path)
    deskew_path = f"{base}_deskewed{ext}"
    cv2.imwrite(deskew_path, result)
    logger.info("Deskewed (%.2f°) → %s", angle, deskew_path)
    return deskew_path


# ---------------------------------------------------------------------------
# PDF extraction — native text + coordinate matrices
# ---------------------------------------------------------------------------
def extract_pdf_data(pdf_path: str) -> dict:
    """
    Blueprint §3.2 — Extract text characters AND coordinate matrices
    from a digitally native PDF.
    Returns a list of page dicts compatible with the stitch heuristic.
    """
    pages_out = []

    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""

            # Coordinate matrix — extract word positions if available
            coords = []
            if hasattr(page, "extract_words"):
                try:
                    words = page.extract_words()
                    coords = [
                        {"text": w.get("text", ""), "x0": w.get("x0"), "y0": w.get("y0"),
                         "x1": w.get("x1"), "y1": w.get("y1")}
                        for w in words
                    ]
                except Exception:
                    pass

            pages_out.append({
                "page_number"    : page_num + 1,
                "extracted_text" : raw_text,
                "coordinates"    : coords,
                "tables"         : [],
                "key_value_pairs": {},
                "handwriting_detected": False,
                "confidence_warning"  : False,
                "confidence_warning_reason": None,
            })

    except Exception as exc:
        logger.error("pypdf extraction failed: %s", exc)

    # Try pypdfium2 for richer binary data if available
    if PDFIUM_AVAILABLE and not any(p["extracted_text"] for p in pages_out):
        try:
            doc = pdfium.PdfDocument(pdf_path)
            for i in range(len(doc)):
                page   = doc.get_page(i)
                textpage = page.get_textpage()
                text   = textpage.get_text_range()
                if i < len(pages_out):
                    pages_out[i]["extracted_text"] = text
                else:
                    pages_out.append({
                        "page_number": i + 1,
                        "extracted_text": text,
                        "coordinates": [],
                        "tables": [],
                        "key_value_pairs": {},
                        "handwriting_detected": False,
                        "confidence_warning": False,
                        "confidence_warning_reason": None,
                    })
        except Exception as exc:
            logger.error("pypdfium2 extraction failed: %s", exc)

    return pages_out


def pdf_has_enough_text(pages: list[dict]) -> bool:
    total = sum(len(p.get("extracted_text", "").strip()) for p in pages)
    return total >= MIN_TEXT_CHARS


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status" : "ok",
        "service": "lego2-triage",
        "pdfium" : PDFIUM_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Triage endpoint
# ---------------------------------------------------------------------------
@app.post("/triage", tags=["Triage"])
async def triage_document(file: UploadFile = File(...)):
    """
    Routing logic:
      • Digital PDF with extractable text → fast_track_complete (no GPU needed)
      • Image or scanned PDF → clean image and return forward_to_vlm
    """
    original_name = file.filename or "unknown"
    _, ext = os.path.splitext(original_name)
    ext = ext.lower()

    dest_path = os.path.join(TEMP_DIR, original_name)
    with open(dest_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    logger.info("Received: %s (ext=%s)", original_name, ext)

    # ── PDF branch ─────────────────────────────────────────────────────────
    if ext == PDF_EXT:
        pages = extract_pdf_data(dest_path)
        if pdf_has_enough_text(pages):
            logger.info("PDF fast-tracked (%d pages, %d chars total)",
                        len(pages), sum(len(p["extracted_text"]) for p in pages))
            return {
                "routing"       : "fast_track_complete",
                "pages"         : pages,
                "extracted_data": pages[0] if pages else {},
            }
        else:
            logger.info("Scanned/blank PDF — forwarding to VLM")
            return {
                "routing"           : "forward_to_vlm",
                "cleaned_file_path" : dest_path,
            }

    # ── Image branch ───────────────────────────────────────────────────────
    elif ext in IMAGE_EXT:
        processed = deskew_image(dest_path)
        processed = apply_glare_suppression(processed)
        return {
            "routing"           : "forward_to_vlm",
            "cleaned_file_path" : processed,
        }

    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: '{ext}'")
