"""
Greencare AI — Lego 4: Human-in-the-Loop Validation Dashboard
=============================================================
Blueprint §3.4 — Complete UI brick with:
  • Upload panel (submits to Lego 1 Gateway, auto-refreshes queue)
  • Side-by-side: original image | AI-extracted JSON (editable)
  • Confidence badges (low-confidence highlight)
  • Editable table grid for manual cell corrections
  • Approve → final_database/  |  Reject → rejected/
  • Export to JSON / CSV / Excel
  • Auto-load first queue item on Refresh

Runs on: Port 7860
"""

import json
import os
import glob
import io
import time
import logging
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
PENDING_DIR  = "./pending_review"
FINAL_DIR    = "./final_database"
REJECTED_DIR = "./rejected"
UPLOAD_DIR   = "./temp_uploads"
LEGO2_TEMP   = "./lego2_temp"
ASSETS_DIR   = "./extracted_assets"
EXPORT_DIR   = "./exports"
GATEWAY_URL  = os.environ.get("GATEWAY_URL", "http://localhost:8000")

for d in [PENDING_DIR, FINAL_DIR, REJECTED_DIR, ASSETS_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def get_pending_files() -> list[str]:
    return sorted(glob.glob(os.path.join(PENDING_DIR, "*.json")))


def get_pending_names() -> list[str]:
    return [os.path.basename(f) for f in get_pending_files()]


def format_stats() -> str:
    p = len(get_pending_files())
    c = len(glob.glob(os.path.join(FINAL_DIR,    "*.json")))
    r = len(glob.glob(os.path.join(REJECTED_DIR, "*.json")))
    return f"**Queue:** {p} pending  |  {c} committed  |  {r} rejected"


def find_source_image(job_id: str) -> str | None:
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    for d in [UPLOAD_DIR, LEGO2_TEMP, ASSETS_DIR]:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.startswith(job_id) and Path(fname).suffix.lower() in exts:
                return os.path.join(d, fname)
    return None


def build_confidence_html(page: dict) -> str:
    warn   = page.get("confidence_warning", False)
    reason = page.get("confidence_warning_reason") or ""
    hw     = page.get("handwriting_detected", False)
    curved = page.get("curved_text_detected", False)

    parts = []
    color = "#ef4444" if warn else "#22c55e"
    label = "LOW CONFIDENCE" if warn else "HIGH CONFIDENCE"
    parts.append(f'<span style="background:{color};color:#fff;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700">{label}</span>')
    if hw:
        parts.append('<span style="background:#f59e0b;color:#fff;padding:3px 10px;border-radius:20px;font-size:12px">Handwriting Detected</span>')
    if curved:
        parts.append('<span style="background:#8b5cf6;color:#fff;padding:3px 10px;border-radius:20px;font-size:12px">Curved Text</span>')

    html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0">' + "".join(parts) + "</div>"
    if warn and reason:
        html += f'<div style="background:#fef2f2;border:1px solid #ef4444;border-radius:8px;padding:8px;font-size:12px;color:#991b1b;margin-top:4px">{reason}</div>'
    return html


def build_first_table_df(data: dict) -> pd.DataFrame:
    pages = data.get("pages", [data])
    for page in pages:
        for tbl in page.get("tables", []):
            rows    = tbl.get("rows", [])
            headers = tbl.get("headers", [])
            if rows:
                try:
                    return pd.DataFrame(rows, columns=headers or None)
                except Exception:
                    pass
    return pd.DataFrame([{"info": "No tables detected in this document"}])


# ---------------------------------------------------------------------------
# Export helpers — use ./exports/ dir (works on Windows, no /tmp)
# ---------------------------------------------------------------------------
def _export_path(job_id: str, ext: str) -> str:
    return os.path.join(EXPORT_DIR, f"{job_id}.{ext}")


def build_export_frames(data: dict) -> dict[str, pd.DataFrame]:
    pages  = data.get("pages", [data])
    merged: dict = {}
    for p in pages:
        merged.update(p)

    frames: dict[str, pd.DataFrame] = {}
    kvp = merged.get("key_value_pairs", {})
    if kvp:
        frames["Key_Value_Pairs"] = pd.DataFrame([{"Field": k, "Value": v} for k, v in kvp.items()])
    text = merged.get("extracted_text", "")
    if text:
        frames["Extracted_Text"] = pd.DataFrame([{"Text": text}])
    for tbl in merged.get("tables", []):
        headers = tbl.get("headers", [])
        rows    = tbl.get("rows",    [])
        if rows:
            idx = tbl.get("table_index", len(frames))
            frames[f"Table_{idx + 1}"] = pd.DataFrame(rows, columns=headers or None)
    return frames


def do_export_json(file_path: str, edited_json: str) -> str | None:
    if not file_path:
        return None
    try:
        data = json.loads(edited_json)
    except Exception:
        return None
    job_id = Path(file_path).stem
    out    = _export_path(job_id, "json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return out


def do_export_excel(file_path: str, edited_json: str) -> str | None:
    if not file_path:
        return None
    try:
        data = json.loads(edited_json)
    except Exception:
        return None
    frames = build_export_frames(data)
    job_id = Path(file_path).stem
    out    = _export_path(job_id, "xlsx")
    buf    = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if not frames:
            pd.DataFrame([{"result": "No structured data"}]).to_excel(writer, sheet_name="Result", index=False)
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    with open(out, "wb") as f:
        f.write(buf.getvalue())
    return out


def do_export_csv(file_path: str, edited_json: str) -> str | None:
    if not file_path:
        return None
    try:
        data = json.loads(edited_json)
    except Exception:
        return None
    frames  = build_export_frames(data)
    job_id  = Path(file_path).stem
    out     = _export_path(job_id, "csv")
    sections = []
    for name, df in frames.items():
        sections.append(f"### {name}\n" + df.to_csv(index=False))
    content = "\n\n".join(sections) if sections else "No structured data"
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    return out


# ---------------------------------------------------------------------------
# Core pipeline actions
# ---------------------------------------------------------------------------
def load_document(file_name: str):
    """
    Load a document from pending_review by filename.
    Returns 8 values: status_md, json_str, image, conf_html,
                       file_path, table_df, text_md, stats
    """
    EMPTY = (
        "Select a document from the queue above.",
        "{}",
        None,
        "",
        None,
        pd.DataFrame([{"info": "No document loaded"}]),
        "",
        format_stats(),
    )

    if not file_name:
        return EMPTY

    file_path = os.path.join(PENDING_DIR, file_name)
    if not os.path.exists(file_path):
        return EMPTY

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to load %s: %s", file_path, exc)
        return EMPTY

    job_id   = data.get("_job_id",   Path(file_path).stem)
    pipeline = data.get("_pipeline", "unknown")
    fname    = data.get("_filename", file_name)
    pages    = data.get("pages",     [data])
    first    = pages[0] if pages else {}
    doc_type = first.get("document_type", "unknown")

    status_md = (
        f"### {fname}\n"
        f"**Job ID:** `{job_id}`   **Pipeline:** `{pipeline}`   "
        f"**Type:** `{doc_type}`   **Pages:** {len(pages)}"
    )
    conf_html  = build_confidence_html(first)
    json_str   = json.dumps(data, indent=4, ensure_ascii=False)
    image_path = find_source_image(job_id)
    table_df   = build_first_table_df(data)

    raw_text    = (first.get("extracted_text", "") or "")[:1200]
    md_tables   = first.get("markdown_tables", "") or ""
    text_md     = f"**Extracted Text:**\n```\n{raw_text}\n```\n\n{md_tables}"

    return (status_md, json_str, image_path, conf_html,
            file_path, table_df, text_md, format_stats())


def refresh_queue():
    """Refresh + auto-load first queued document. Returns all display outputs."""
    names = get_pending_names()
    dd_update = gr.update(choices=names, value=names[0] if names else None)
    if not names:
        return (
            dd_update, format_stats(),
            "No documents in queue. Upload one above.",
            "{}", None, "",
            None,
            pd.DataFrame([{"info": "Queue is empty"}]),
            "",
        )
    status, js, img, conf, fpath, tbl, txt, stats = load_document(names[0])
    return (dd_update, stats, status, js, img, conf, fpath, tbl, txt)


def approve_document(file_path: str, edited_json: str):
    if not file_path or not os.path.exists(file_path):
        return "No document loaded — nothing to approve.", format_stats()
    try:
        final = json.loads(edited_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON — fix the error before approving:\n`{e}`", format_stats()

    final["_approved_at"]  = datetime.utcnow().isoformat() + "Z"
    final["_reviewed_by"] = "human_reviewer"

    dest = os.path.join(FINAL_DIR, os.path.basename(file_path))
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    os.remove(file_path)
    logger.info("Approved: %s", dest)
    return "Committed! Record saved to final_database/.", format_stats()


def reject_document(file_path: str, reason: str):
    if not file_path or not os.path.exists(file_path):
        return "No document loaded.", format_stats()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    data["_rejected_at"]      = datetime.utcnow().isoformat() + "Z"
    data["_rejection_reason"] = reason or "No reason"

    dest = os.path.join(REJECTED_DIR, os.path.basename(file_path))
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.remove(file_path)
    logger.info("Rejected: %s", dest)
    return "Rejected. Moved to rejected/.", format_stats()


def upload_and_process(file_obj):
    """Upload file to gateway, wait for processing, then refresh queue."""
    if file_obj is None:
        return ("No file selected.",) + _empty_refresh()

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    filename  = os.path.basename(file_path)

    try:
        with open(file_path, "rb") as fh:
            resp = requests.post(
                f"{GATEWAY_URL}/api/v1/ingest",
                files={"file": (filename, fh)},
                timeout=30,
            )
        resp.raise_for_status()
        job_id = resp.json().get("job_id", "?")
        upload_msg = f"Submitted! Job ID: `{job_id}` — waiting for processing..."
    except requests.ConnectionError:
        return (
            "Cannot reach the API Gateway at localhost:8000. Make sure run_all.py is running.",
        ) + _empty_refresh()
    except Exception as exc:
        return (f"Upload failed: {exc}",) + _empty_refresh()

    # Wait for the background task to write pending_review JSON
    # Poll up to 30 seconds
    expected = os.path.join(PENDING_DIR, f"{job_id}.json")
    for _ in range(30):
        if os.path.exists(expected):
            break
        time.sleep(1)

    result = refresh_queue()
    return (upload_msg,) + result


def _empty_refresh():
    names = get_pending_names()
    dd    = gr.update(choices=names, value=None)
    return (
        dd, format_stats(),
        "Select a document from the queue.", "{}", None, "",
        None,
        pd.DataFrame([{"info": "No document loaded"}]),
        "",
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Greencare AI HITL Dashboard") as dashboard:

    current_file = gr.State(value=None)

    # Header
    gr.Markdown("# Greencare AI — Human Review Dashboard\n*Blueprint §3.4 — Review, correct, and approve AI extractions.*")
    stats_bar = gr.Markdown(value=format_stats())

    # ── STEP 1: Upload ──────────────────────────────────────────────────────
    with gr.Accordion("Step 1 — Upload Document", open=True):
        gr.Markdown(
            "Select a file (PDF, JPG, PNG, TIFF, BMP). It will be sent through the full AI pipeline automatically. "
            "The document will appear in the review queue below once processed (~5–15 seconds)."
        )
        with gr.Row():
            upload_widget = gr.File(
                label="Choose File",
                file_types=[".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"],
                scale=3,
            )
            submit_btn = gr.Button("Submit to AI Pipeline", variant="primary", scale=1, size="lg")
        upload_msg_box = gr.Markdown(value="")

    # ── STEP 2: Review Queue ────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### Step 2 — Select Document from Queue to Review")

    with gr.Row():
        queue_dropdown = gr.Dropdown(
            label="Pending Documents",
            choices=get_pending_names(),
            interactive=True,
            scale=4,
        )
        refresh_btn = gr.Button("Refresh Queue", variant="secondary", scale=1)

    doc_status = gr.Markdown(value="*Upload a document or click Refresh to load the queue.*")

    with gr.Row():
        conf_box = gr.HTML(value="")

    # Side-by-side: image + JSON
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("**Original Document**")
            img_view = gr.Image(label="Source Image", interactive=False, height=520)

        with gr.Column(scale=1):
            gr.Markdown("**AI Extracted JSON — edit to fix errors**")
            json_view = gr.Code(label="JSON", language="json", lines=26, interactive=True)

    # Table grid
    with gr.Accordion("Extracted Table (Editable Grid)", open=False):
        table_grid = gr.Dataframe(label="Table Data", interactive=True, wrap=True)

    # Text preview
    with gr.Accordion("Extracted Text & Markdown Tables", open=False):
        text_view = gr.Markdown(value="")

    # ── STEP 3: Approve / Reject ────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### Step 3 — Approve or Reject")

    with gr.Row():
        approve_btn = gr.Button("Approve & Commit to Database", variant="primary", scale=2)
        reject_btn  = gr.Button("Reject Document", variant="stop", scale=1)

    rejection_box = gr.Textbox(
        label="Rejection Reason (fill in before rejecting)",
        placeholder="e.g. Wrong document type, illegible scan, duplicate entry...",
    )
    action_result = gr.Markdown(value="")

    # ── STEP 4: Export ──────────────────────────────────────────────────────
    with gr.Accordion("Step 4 — Export Data", open=False):
        gr.Markdown("Generate a download file from the current JSON (before or after approval).")
        with gr.Row():
            btn_json  = gr.Button("Generate JSON",  scale=1)
            btn_excel = gr.Button("Generate Excel", scale=1)
            btn_csv   = gr.Button("Generate CSV",   scale=1)
        export_out = gr.File(label="Download", interactive=False)

    # ── Event wiring ────────────────────────────────────────────────────────

    # Shared output list for refresh/load operations
    REFRESH_OUTPUTS = [
        queue_dropdown, stats_bar,
        doc_status, json_view, img_view, conf_box,
        current_file,
        table_grid, text_view,
    ]

    def on_dropdown_change(selected_name):
        if not selected_name:
            return (
                format_stats(),
                "Select a document from the queue.",
                "{}", None, "",
                None,
                pd.DataFrame([{"info": "No document loaded"}]),
                "",
            )
        status, js, img, conf, fpath, tbl, txt, stats = load_document(selected_name)
        return stats, status, js, img, conf, fpath, tbl, txt

    submit_btn.click(
        fn=upload_and_process,
        inputs=[upload_widget],
        outputs=[upload_msg_box] + REFRESH_OUTPUTS,
    )

    refresh_btn.click(
        fn=refresh_queue,
        outputs=REFRESH_OUTPUTS,
    )

    queue_dropdown.change(
        fn=on_dropdown_change,
        inputs=[queue_dropdown],
        outputs=[
            stats_bar, doc_status, json_view, img_view, conf_box,
            current_file, table_grid, text_view,
        ],
    )

    approve_btn.click(
        fn=approve_document,
        inputs=[current_file, json_view],
        outputs=[action_result, stats_bar],
    )

    reject_btn.click(
        fn=reject_document,
        inputs=[current_file, rejection_box],
        outputs=[action_result, stats_bar],
    )

    btn_json.click(
        fn=do_export_json,
        inputs=[current_file, json_view],
        outputs=[export_out],
    )
    btn_excel.click(
        fn=do_export_excel,
        inputs=[current_file, json_view],
        outputs=[export_out],
    )
    btn_csv.click(
        fn=do_export_csv,
        inputs=[current_file, json_view],
        outputs=[export_out],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="emerald",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    )
