# 🌿 Greencare AI — Groq-Powered Intelligent Document Processing

A production-ready, four-microservice IDP pipeline that uses **Groq's Llama-3.2 Vision** to extract structured JSON from any document (PDFs, scanned images, handwritten forms) at blazing speed — no local GPU required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (HTTP POST)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │  POST /api/v1/ingest
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 1 — FastAPI Gateway  (port 8000)                           │
│  • UUID job assignment                                           │
│  • File persistence to shared volume                             │
│  • Celery task enqueue → Redis                                   │
└──────────────────────────────┬───────────────────────────────────┘
                               │  Celery Worker pulls task
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 2 — CPU Triage  (port 8001)                                │
│  ┌─────────────────┐         ┌──────────────────────────────┐   │
│  │  Digital PDF    │──Text──▶│  Fast-Track (no Groq token)  │   │
│  └─────────────────┘         └──────────────────────────────┘   │
│  ┌─────────────────┐                                            │
│  │  Image / Scanned│──Deskew + Glare-Suppress ──────────────▶  │
│  └─────────────────┘                                            │
└──────────────────────────────┬───────────────────────────────────┘
                               │  image forwarded
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 3 — Groq Vision Engine  (port 8002)                        │
│  • Base64 image encoding                                         │
│  • llama-3.2-90b-vision-preview @ temperature=0.0               │
│  • response_format=json_object (grammar-locked)                  │
│  • Retry + exponential back-off on rate limits                   │
└──────────────────────────────┬───────────────────────────────────┘
                               │  JSON written to pending_review/
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 4 — HITL Dashboard  (port 7860)                            │
│  • Gradio web UI                                                 │
│  • Queue navigation, original image preview                      │
│  • Editable JSON panel                                           │
│  • Approve → final_database/ | Reject → rejected/               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- A [Groq API key](https://console.groq.com/keys) (free tier available)

### 1. Clone / navigate to the project
```bash
cd greencare-ai
```

### 2. Set your API key
```bash
# Windows PowerShell
Copy-Item .env.example .env
# Then open .env in a text editor and paste your GROQ_API_KEY
```

```bash
# Linux / macOS
cp .env.example .env
nano .env   # paste GROQ_API_KEY=gsk_...
```

### 3. Build and start all services
```bash
docker-compose up --build
```

First build takes ~3-5 minutes (downloading Python image + deps). Subsequent starts are instant.

### 4. Open the services

| Service | URL |
|---|---|
| 🚪 API Gateway (Swagger UI) | http://localhost:8000/docs |
| 🔍 CPU Triage (Swagger UI) | http://localhost:8001/docs |
| 🤖 Groq Vision (Swagger UI) | http://localhost:8002/docs |
| 👁 HITL Dashboard | http://localhost:7860 |

---

## Running Without Docker (Development Mode)

Install dependencies:
```bash
pip install -r requirements.txt
```

Start Redis (use Docker just for Redis):
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

Open four terminals and run:

**Terminal 1 — Lego 2 (Triage)**
```bash
uvicorn lego2_triage.triage_service:app --port 8001 --reload
```

**Terminal 2 — Lego 3 (Groq Engine)**
```bash
set GROQ_API_KEY=gsk_your_key_here   # Windows
uvicorn lego3_groq.groq_engine:app --port 8002 --reload
```

**Terminal 3 — Celery Worker**
```bash
celery -A lego1_gateway.worker.celery_app worker --loglevel=info
```

**Terminal 4 — Lego 1 (Gateway) + Lego 4 (HITL)**
```bash
uvicorn lego1_gateway.main:app --port 8000 --reload
# In a separate terminal:
python lego4_hitl/hitl_ui.py
```

---

## Testing the Pipeline

### Submit a digital PDF (fast-track, no Groq token used)
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/your/document.pdf"
```

### Submit a photo/scan (will call Groq)
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/scanned_form.jpg"
```

### Check job status
```bash
curl http://localhost:8000/api/v1/status/{job_id}
```

### Open the HITL dashboard to review results
Navigate to **http://localhost:7860**, click **Refresh Queue**, then navigate through pending documents.

---

## Extracted JSON Schema

Every document processed through Groq returns:

```json
{
  "document_type": "invoice",
  "language": "en",
  "extracted_text": "Full verbatim text...",
  "key_value_pairs": {
    "Invoice No": "INV-2024-001",
    "Date": "2024-01-15",
    "Total Amount": "$1,250.00"
  },
  "tables": [
    {
      "table_index": 0,
      "headers": ["Item", "Qty", "Price"],
      "rows": [["Widget A", "10", "$125.00"]]
    }
  ],
  "handwriting_detected": false,
  "confidence_warning": false,
  "confidence_warning_reason": null,
  "_pipeline": "groq_vision",
  "_job_id": "uuid-here"
}
```

---

## Trade-offs & Design Decisions

| Decision | Rationale |
|---|---|
| Groq API over local Qwen | 200x faster inference, zero GPU cost, no TensorRT compilation |
| `response_format=json_object` | Grammar-level enforcement prevents hallucinated JSON formats |
| `temperature=0.0` | Deterministic extraction; identical docs produce identical outputs |
| CPU triage first | Saves Groq tokens for digital PDFs; ~60% of enterprise docs are digital |
| OpenCV deskew + CLAHE | Better image quality → fewer VLM errors and lower confidence warnings |
| HITL before database write | Human review catches the ~5-10% edge cases where VLMs err |

> ⚠️ **Data Privacy Note:** This architecture sends document content to Groq's cloud API. For documents containing PII or PHI, ensure your use complies with Groq's [Terms of Service](https://groq.com/terms-of-use/) and applicable privacy regulations (GDPR, HIPAA).

---

## Project Structure

```
greencare-ai/
├── Dockerfile                  # Unified image for all services
├── docker-compose.yml          # Orchestrates all 6 containers
├── requirements.txt            # Python dependencies
├── .env.example                # API key template
├── README.md                   # This file
│
├── lego1_gateway/
│   ├── main.py                 # FastAPI Gateway (port 8000)
│   └── worker.py               # Celery pipeline orchestrator
│
├── lego2_triage/
│   └── triage_service.py       # CPU fast-track + image enhancement (port 8001)
│
├── lego3_groq/
│   └── groq_engine.py          # Groq Llama-3.2 Vision extraction (port 8002)
│
└── lego4_hitl/
    └── hitl_ui.py              # Gradio HITL dashboard (port 7860)
```
