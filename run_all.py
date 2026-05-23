"""
Greencare AI — Master Launcher (No Docker, No Redis Required)
=============================================================
Starts all 4 Lego services in separate background processes:
  Lego 2 → http://localhost:8001  (CPU Triage)
  Lego 3 → http://localhost:8002  (Groq Vision Engine)
  Lego 1 → http://localhost:8000  (API Gateway + Serialization)
  Lego 4 → http://localhost:7860  (HITL Dashboard)

Usage:
  python run_all.py
"""

import os
import sys
import time
import threading
import subprocess
import webbrowser
import urllib.request
import urllib.error

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Load .env ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("❌  GROQ_API_KEY not set. Please add it to .env")
    sys.exit(1)

PYTHON = sys.executable

# ── Service definitions ────────────────────────────────────────────────────
SERVICES = [
    {
        "name"   : "Lego2-CPU-Triage",
        "cmd"    : [PYTHON, "-m", "uvicorn", "lego2_triage.triage_service:app",
                    "--host", "0.0.0.0", "--port", "8001"],
        "health" : "http://localhost:8001/health",
        "color"  : "\033[94m",
    },
    {
        "name"   : "Lego3-Groq-Engine",
        "cmd"    : [PYTHON, "-m", "uvicorn", "lego3_groq.groq_engine:app",
                    "--host", "0.0.0.0", "--port", "8002"],
        "health" : "http://localhost:8002/health",
        "color"  : "\033[95m",
    },
    {
        "name"   : "Lego1-API-Gateway",
        "cmd"    : [PYTHON, "-m", "uvicorn", "lego1_gateway.main_standalone:app",
                    "--host", "0.0.0.0", "--port", "8000"],
        "health" : "http://localhost:8000/health",
        "color"  : "\033[92m",
    },
    {
        "name"   : "Lego4-HITL-UI",
        "cmd"    : [PYTHON, "lego4_hitl/hitl_ui.py"],
        "health" : "http://localhost:7860",
        "color"  : "\033[93m",
    },
]

RESET = "\033[0m"
BOLD  = "\033[1m"
processes: list[subprocess.Popen] = []


def wait_for(url: str, timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1.5)
    return False


def stream_output(proc: subprocess.Popen, name: str, color: str):
    for line in proc.stdout:
        sys.stdout.write(f"{color}[{name}]{RESET} {line}")
        sys.stdout.flush()


def launch_service(svc: dict):
    env = os.environ.copy()
    proc = subprocess.Popen(
        svc["cmd"],
        cwd=BASE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    processes.append(proc)
    threading.Thread(
        target=stream_output,
        args=(proc, svc["name"], svc["color"]),
        daemon=True,
    ).start()
    return proc


def main():
    print(f"\n{BOLD}[*] Greencare AI -- Starting All Services{RESET}")
    print(f"{'=' * 55}")

    # Create required directories
    for d in ["temp_uploads", "pending_review", "final_database",
              "rejected", "lego2_temp", "extracted_assets"]:
        os.makedirs(os.path.join(BASE, d), exist_ok=True)

    # Start Lego 2 & 3 first (gateway depends on them)
    for svc in SERVICES[:2]:
        print(f"\n[..] Starting {svc['name']}...")
        launch_service(svc)
        if wait_for(svc["health"]):
            print(f"[OK] {svc['name']} ready.")
        else:
            print(f"[!!] {svc['name']} didn't respond in time -- continuing anyway.")

    # Then start Gateway and HITL
    for svc in SERVICES[2:]:
        print(f"\n[..] Starting {svc['name']}...")
        launch_service(svc)
        time.sleep(3)

    print(f"\n{BOLD}{'=' * 55}{RESET}")
    print(f"{BOLD}[LIVE] Greencare AI is running!{RESET}")
    print(f"  [1] API Gateway    --> http://localhost:8000/docs")
    print(f"  [2] CPU Triage     --> http://localhost:8001/docs")
    print(f"  [3] Groq Engine    --> http://localhost:8002/docs")
    print(f"  [4] HITL Dashboard --> http://localhost:7860")
    print(f"{BOLD}{'=' * 55}{RESET}")
    print("\nPress Ctrl+C to stop all services.\n")

    def open_browser():
        time.sleep(10)
        webbrowser.open("http://localhost:8000/docs")
        time.sleep(3)
        webbrowser.open("http://localhost:7860")
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        while True:
            time.sleep(1)
            # Restart crashed services
            for i, (svc, proc) in enumerate(zip(SERVICES, processes)):
                if proc.poll() is not None:
                    print(f"\n[!!] {svc['name']} crashed (exit {proc.returncode}) -- restarting...")
                    new_proc = launch_service(svc)
                    processes[i] = new_proc
    except KeyboardInterrupt:
        print(f"\n{BOLD}[STOP] Shutting down all services...{RESET}")
        for p in processes:
            p.terminate()
        print("All services stopped. Goodbye!")


if __name__ == "__main__":
    main()
