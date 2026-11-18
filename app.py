#!/usr/bin/env python3
"""
ClipMore Studio - Modern Web UI Backend
---------------------------------------
FastAPI server providing a web dashboard for autonomous AI stream clipping
and secondary fallback DSP clipping, with live logs and video previews.
"""

import os
import sys
import time
import signal
import asyncio
import subprocess
import threading
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure FFmpeg and NVIDIA CUDA paths are bootstraped
PROJECT_ROOT = Path(__file__).resolve().parent
AI_DIR = PROJECT_ROOT / "ai"
MANUAL_DIR = PROJECT_ROOT / "manual"
EXPORTS_DIR = PROJECT_ROOT / "exports"
FALLBACK_EXPORTS_DIR = PROJECT_ROOT / "fallback_exports"
RAW_CHUNKS_DIR = PROJECT_ROOT / "raw_chunks"
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"

for d in (EXPORTS_DIR, FALLBACK_EXPORTS_DIR, RAW_CHUNKS_DIR, TEMPLATES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Bootstrap CUDA and FFmpeg DLLs
def bootstrap_env():
    venv_nvidia = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "nvidia"
    if venv_nvidia.exists():
        for sub in venv_nvidia.iterdir():
            b = sub / "bin"
            if b.is_dir() and str(b) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(b))
                    except OSError:
                        pass
    for base in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages",
        Path("C:/ProgramData/chocolatey/bin"),
    ]:
        if base.exists():
            for exe in base.glob("**/ffmpeg.exe"):
                bin_dir = str(exe.parent)
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                return

bootstrap_env()

app = FastAPI(title="ClipMore Studio", version="1.0")

# ---------------------------------------------------------------------------
# Global Process State
# ---------------------------------------------------------------------------
class ClipperState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.mode: str = "idle"  # "ai", "fallback", "test", "idle"
        self.current_url: str = ""
        self.start_time: float = 0.0
        self.log_history: List[str] = []
        self.lock = threading.Lock()

    def append_log(self, text: str):
        with self.lock:
            self.log_history.append(text)
            if len(self.log_history) > 800:
                self.log_history = self.log_history[-800:]

    def clear_logs(self):
        with self.lock:
            self.log_history.clear()

state = ClipperState()


def log_reader_thread(proc: subprocess.Popen):
    """Continuously reads stdout and stderr from clipper process."""
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            clean = line.strip()
            if clean:
                state.append_log(clean)
    except Exception:
        pass
    finally:
        state.mode = "idle"
        state.append_log("[SYSTEM] Clipper process terminated.")


def get_gpu_info() -> dict:
    """Queries nvidia-smi for real-time GPU stats."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
               "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 5:
                return {
                    "available": True,
                    "name": parts[0],
                    "memory_total": f"{parts[1]} MB",
                    "memory_used": f"{parts[2]} MB",
                    "utilization": f"{parts[3]}%",
                    "temp": f"{parts[4]}°C"
                }
    except Exception:
        pass
    return {"available": False, "name": "NVIDIA GPU Not Detected", "memory_total": "N/A", "memory_used": "N/A", "utilization": "0%", "temp": "N/A"}


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------
class StartAIRequest(BaseModel):
    url: str
    threshold: float = 2.5
    model: str = "small.en"


class StartFallbackRequest(BaseModel):
    file_path: str
    max_clips: int = 5


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index template not found.")
    return index_file.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    uptime = int(time.time() - state.start_time) if state.mode != "idle" else 0
    gpu = get_gpu_info()
    
    # Count generated clips
    ai_clips = len(list(EXPORTS_DIR.glob("*.mp4")))
    fallback_clips = len(list(FALLBACK_EXPORTS_DIR.glob("*.mp4")))
    raw_chunks = len(list(RAW_CHUNKS_DIR.glob("*.ts")))

    return {
        "mode": state.mode,
        "is_running": state.mode != "idle",
        "current_url": state.current_url,
        "uptime_seconds": uptime,
        "gpu": gpu,
        "clip_counts": {
            "ai_clips": ai_clips,
            "fallback_clips": fallback_clips,
            "total_clips": ai_clips + fallback_clips,
            "active_chunks": raw_chunks
        }
    }


@app.post("/api/ai/start")
async def start_ai_clipper(req: StartAIRequest):
    if state.mode != "idle":
        raise HTTPException(status_code=400, detail="A clipping process is already running. Please stop it first.")
    
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please enter a valid live stream URL.")

    script_path = AI_DIR / "auto_viral_clipper.py"
    python_exe = sys.executable

    state.clear_logs()
    state.mode = "ai"
    state.current_url = url
    state.start_time = time.time()
    state.append_log(f"[SYSTEM] Launching Autonomous AI Clipper for: {url}")

    cmd = [python_exe, str(script_path), url]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT)
    )
    state.process = proc

    t = threading.Thread(target=log_reader_thread, args=(proc,), daemon=True)
    t.start()

    return {"status": "started", "mode": "ai", "url": url}


@app.post("/api/fallback/start")
async def start_fallback_clipper(req: StartFallbackRequest):
    if state.mode != "idle":
        raise HTTPException(status_code=400, detail="A clipping process is already running. Please stop it first.")

    target_file = Path(req.file_path.strip().strip('"'))
    if not target_file.exists():
        raise HTTPException(status_code=400, detail=f"Target file does not exist: {target_file}")

    script_path = MANUAL_DIR / "fallback_clipper.py"
    python_exe = sys.executable

    state.clear_logs()
    state.mode = "fallback"
    state.current_url = str(target_file.name)
    state.start_time = time.time()
    state.append_log(f"[SYSTEM] Launching Fallback DSP Clipper on: {target_file.name}")

    cmd = [python_exe, str(script_path), str(target_file), "--max-clips", str(req.max_clips)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT)
    )
    state.process = proc

    t = threading.Thread(target=log_reader_thread, args=(proc,), daemon=True)
    t.start()

    return {"status": "started", "mode": "fallback", "file": str(target_file.name)}


@app.post("/api/test")
async def run_pipeline_test():
    if state.mode != "idle":
        raise HTTPException(status_code=400, detail="A clipping process is currently active.")

    state.clear_logs()
    state.mode = "test"
    state.current_url = "Self-Test Verification"
    state.start_time = time.time()
    state.append_log("[SYSTEM] Starting end-to-end self-test verification...")

    cmd = [sys.executable, str(AI_DIR / "auto_viral_clipper.py"), "--test"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT)
    )
    state.process = proc

    t = threading.Thread(target=log_reader_thread, args=(proc,), daemon=True)
    t.start()

    return {"status": "testing"}


@app.post("/api/stop")
async def stop_clipper():
    if state.process and state.process.poll() is None:
        state.append_log("[SYSTEM] User sent stop signal. Terminating process tree...")
        try:
            # Kill process tree on Windows
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(state.process.pid)], capture_output=True)
        except Exception as e:
            state.process.kill()
        state.mode = "idle"
        state.append_log("[SYSTEM] Process stopped successfully.")
        return {"status": "stopped"}
    state.mode = "idle"
    return {"status": "already_idle"}


@app.get("/api/logs")
async def get_logs(last_index: int = Query(0)):
    with state.lock:
        logs_slice = state.log_history[last_index:]
        next_index = len(state.log_history)
    return {
        "logs": logs_slice,
        "next_index": next_index,
        "is_running": state.mode != "idle"
    }


@app.get("/api/clips")
async def list_clips():
    clips = []
    
    # 1. AI clips from exports/
    for p in sorted(EXPORTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True):
        stat = p.stat()
        name = p.stem
        # Extract score & hook from name if present
        # Format: clip_YYYYMMDD_HHMMSS_scoreXX_HOOK_NAME.mp4
        score = None
        hook = "Viral Clip"
        parts = name.split("_")
        for i, pt in enumerate(parts):
            if pt.startswith("score"):
                try:
                    score = int(pt.replace("score", ""))
                except Exception:
                    pass
                hook = " ".join(parts[i+1:]) if i + 1 < len(parts) else "Viral Moment"
                break
        
        clips.append({
            "folder": "exports",
            "filename": p.name,
            "title": hook.replace("_", " ").upper(),
            "score": score or 90,
            "type": "AI Autonomous",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "url": f"/api/video/exports/{p.name}"
        })

    # 2. Fallback clips from fallback_exports/
    for p in sorted(FALLBACK_EXPORTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True):
        stat = p.stat()
        clips.append({
            "folder": "fallback_exports",
            "filename": p.name,
            "title": f"DSP Highlight ({p.stem})",
            "score": 85,
            "type": "Fallback DSP",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "url": f"/api/video/fallback_exports/{p.name}"
        })

    return {"clips": clips}


@app.get("/api/video/{folder}/{filename}")
async def stream_video(folder: str, filename: str):
    if folder not in ("exports", "fallback_exports"):
        raise HTTPException(status_code=400, detail="Invalid folder")
    base = EXPORTS_DIR if folder == "exports" else FALLBACK_EXPORTS_DIR
    target = base / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path=str(target), media_type="video/mp4", filename=filename)


@app.post("/api/open-folder")
async def open_exports_folder(folder: str = Query("exports")):
    target = EXPORTS_DIR if folder == "exports" else FALLBACK_EXPORTS_DIR
    try:
        os.startfile(str(target))
        return {"status": "opened", "path": str(target)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" 🚀 ClipMore Studio UI Starting at: http://127.0.0.1:8501")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
