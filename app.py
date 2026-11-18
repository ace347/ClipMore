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
MANUAL_EXPORTS_DIR = PROJECT_ROOT / "manual_exports"
RAW_CHUNKS_DIR = PROJECT_ROOT / "raw_chunks"
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"

for d in (EXPORTS_DIR, FALLBACK_EXPORTS_DIR, MANUAL_EXPORTS_DIR, RAW_CHUNKS_DIR, TEMPLATES_DIR):
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


CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_last_gpu_check = 0.0
_cached_gpu_info = {"available": False, "name": "NVIDIA GPU Not Detected", "memory_total": "N/A", "memory_used": "N/A", "utilization": "0%", "temp": "N/A"}

def get_gpu_info() -> dict:
    """Queries nvidia-smi for real-time GPU stats (cached for 4 seconds, silent background execution)."""
    global _last_gpu_check, _cached_gpu_info
    now = time.time()
    if now - _last_gpu_check < 4.0:
        return _cached_gpu_info
    _last_gpu_check = now

    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
               "--format=csv,noheader,nounits"]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
            creationflags=CREATE_NO_WINDOW
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 5:
                _cached_gpu_info = {
                    "available": True,
                    "name": parts[0],
                    "memory_total": f"{parts[1]} MB",
                    "memory_used": f"{parts[2]} MB",
                    "utilization": f"{parts[3]}%",
                    "temp": f"{parts[4]}°C"
                }
                return _cached_gpu_info
    except Exception:
        pass
    return _cached_gpu_info


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


class ManualClipRequest(BaseModel):
    file_path: str
    start: float          # seconds
    end: float            # seconds
    layout: str = "split"  # "split" | "center" | "blur" | "original"
    name: str = ""        # optional label added to the output filename


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
    manual_clips = len(list(MANUAL_EXPORTS_DIR.glob("*.mp4")))
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
            "manual_clips": manual_clips,
            "total_clips": ai_clips + fallback_clips + manual_clips,
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
        cwd=str(PROJECT_ROOT),
        creationflags=CREATE_NO_WINDOW
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
        cwd=str(PROJECT_ROOT),
        creationflags=CREATE_NO_WINDOW
    )
    state.process = proc

    t = threading.Thread(target=log_reader_thread, args=(proc,), daemon=True)
    t.start()

    return {"status": "started", "mode": "fallback", "file": str(target_file.name)}


# ---------------------------------------------------------------------------
# Manual clip: choose a section yourself and export it as a 9:16 clip
# ---------------------------------------------------------------------------
import re

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".ts", ".flv", ".avi", ".m4v"}
MAX_MANUAL_CLIP_SECONDS = 600
_manual_lock = threading.Lock()

MANUAL_FILTERS = {
    # Webcam strip on top, gameplay below (same look as the DSP clipper).
    # The source is normalised to 1920x1080 first so the fixed crops always fit.
    "split": (
        "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,split=2[s1][s2];"
        "[s1]crop=480:360:20:700,scale=1080:640:force_original_aspect_ratio=increase,crop=1080:640[cam];"
        "[s2]crop=1080:1080:(in_w-1080)/2:0,scale=1080:1280[game];"
        "[cam][game]vstack=inputs=2[v]"
    ),
    # Plain centre crop to 9:16.
    "center": "[0:v]crop=ih*9/16:ih,scale=1080:1920,setsar=1[v]",
    # Whole frame kept, fitted to the width over a blurred copy of itself.
    "blur": (
        "[0:v]split=2[a][b];"
        "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
        "[b]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
    ),
    # No reframing: keep the original shape, just trim.
    "original": "[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v]",
}

_nvenc_cache: Optional[bool] = None

def nvenc_available() -> bool:
    global _nvenc_cache
    if _nvenc_cache is None:
        try:
            _nvenc_cache = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.2:r=30",
                 "-pix_fmt", "yuv420p", "-c:v", "h264_nvenc", "-f", "null", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8,
                creationflags=CREATE_NO_WINDOW,
            ).returncode == 0
        except Exception:
            _nvenc_cache = False
    return _nvenc_cache


def fmt_ts(seconds: float) -> str:
    s = max(0.0, seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:04.1f}" if h else f"{int(m)}:{sec:04.1f}"


def run_manual_export(req: ManualClipRequest, in_args: list, out_args: Optional[list] = None) -> Path:
    """Cuts [start, end] using the given ffmpeg input args, reframes it, and writes it to manual_exports/.

    in_args  - ffmpeg input side (e.g. ["-ss", .., "-t", .., "-i", file])
    out_args - extra output-side options (used for the live recording's segment offset)
    """
    duration = req.end - req.start
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", req.name).strip("_")[:40]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"manual_{stamp}_{int(req.start)}s-{int(req.end)}s" + (f"_{label}" if label else "") + ".mp4"
    out_path = MANUAL_EXPORTS_DIR / out_name

    def build(encoder: str, preset: str, quality: list) -> list:
        return [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *in_args,
            "-filter_complex", MANUAL_FILTERS[req.layout],
            "-map", "[v]", "-map", "0:a?",
            *(out_args or []),
            "-c:v", encoder, "-preset", preset, *quality,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", str(out_path),
        ]

    attempts = []
    if nvenc_available():
        attempts.append(("h264_nvenc", build("h264_nvenc", "p4", ["-cq", "23"])))
    attempts.append(("libx264", build("libx264", "fast", ["-crf", "20"])))

    last_err = ""
    for encoder, cmd in attempts:
        state.append_log(f"[EXPORT] Encoding with {encoder}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, creationflags=CREATE_NO_WINDOW)
        if res.returncode == 0 and out_path.exists():
            return out_path
        last_err = (res.stderr or "").strip().splitlines()[-1:] or ["unknown ffmpeg error"]
        last_err = last_err[0]
        state.append_log(f"[ERROR] {encoder} failed: {last_err}")
    raise RuntimeError(last_err)


@app.get("/api/manual/source")
async def manual_source(path: str = Query(...)):
    """Streams a local video (with range support) so the browser can preview and scrub it."""
    p = Path(path.strip().strip('"'))
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return FileResponse(path=str(p))


@app.post("/api/manual/browse")
async def manual_browse():
    """Opens a native file-picker on the local machine and returns the chosen path."""
    def _pick() -> str:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            return filedialog.askopenfilename(
                title="Choose a video to clip",
                filetypes=[("Video files", "*.mp4 *.mkv *.mov *.webm *.ts *.flv *.avi *.m4v"),
                           ("All files", "*.*")],
            )
        finally:
            root.destroy()
    try:
        path = await asyncio.to_thread(_pick)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File picker unavailable ({e}). Paste the path instead.")
    return {"path": path or ""}


@app.post("/api/manual/export")
async def manual_export(req: ManualClipRequest):
    src = Path(req.file_path.strip().strip('"'))
    if not src.is_file():
        raise HTTPException(status_code=400, detail=f"Video file does not exist: {src}")
    if req.layout not in MANUAL_FILTERS:
        raise HTTPException(status_code=400, detail=f"Unknown layout: {req.layout}")
    if req.start < 0 or req.end <= req.start:
        raise HTTPException(status_code=400, detail="The end time must be after the start time.")
    if req.end - req.start > MAX_MANUAL_CLIP_SECONDS:
        raise HTTPException(status_code=400, detail=f"Clips are limited to {MAX_MANUAL_CLIP_SECONDS // 60} minutes.")
    if not _manual_lock.acquire(blocking=False):
        raise HTTPException(status_code=400, detail="A manual export is already running. Please wait for it to finish.")

    try:
        state.append_log(
            f"[EXPORT] Manual clip {fmt_ts(req.start)} -> {fmt_ts(req.end)} "
            f"({req.end - req.start:.1f}s) from {src.name} [{req.layout}]"
        )
        in_args = ["-ss", f"{req.start:.3f}", "-t", f"{req.end - req.start:.3f}", "-i", str(src)]
        out = await asyncio.to_thread(run_manual_export, req, in_args)
    except Exception as e:
        state.append_log(f"[ERROR] Manual export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
    finally:
        _manual_lock.release()

    state.append_log(f"[EXPORT] FINISHED -> manual_exports/{out.name}")
    return {
        "status": "done",
        "filename": out.name,
        "url": f"/api/video/manual_exports/{out.name}",
        "size_mb": round(out.stat().st_size / (1024 * 1024), 2),
    }


# ---------------------------------------------------------------------------
# Live manual clip: record a live link into a seekable buffer, clip any moment
# ---------------------------------------------------------------------------
import atexit
import shutil
from urllib.parse import urljoin

LIVE_DIR = PROJECT_ROOT / "live_sessions"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
MIN_FREE_BYTES = 3 * 1024 ** 3  # stop recording if the disk drops below 3 GB free

# ffmpeg HLS "event" playlist: it only ever grows, so the player can seek from
# the very first recorded second up to the newest segment.
HLS_OUT = [
    "-f", "hls", "-hls_time", "4", "-hls_list_size", "0",
    "-hls_playlist_type", "event",
    "-hls_flags", "independent_segments+temp_file",
    "-hls_segment_filename", "seg_%05d.ts", "index.m3u8",
]


class LiveClipRequest(BaseModel):
    start: float
    end: float
    layout: str = "split"
    name: str = ""


class LiveConnectRequest(BaseModel):
    url: str
    backfill: str = "start"   # "start" | "now" | number of seconds back


class LiveStopRequest(BaseModel):
    delete: bool = False


def is_youtube_url(u: str) -> bool:
    u = u.lower()
    return "youtube.com" in u or "youtu.be" in u


def fmt_hms(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


class LiveSession:
    def __init__(self, url: str, backfill: str):
        self.id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.url = url
        self.backfill = backfill
        self.dir = LIVE_DIR / self.id
        self.platform = "youtube" if is_youtube_url(url) else "streamlink"
        self.procs: List[subprocess.Popen] = []
        self.status = "connecting"   # connecting | recording | ended | stopped | error
        self.error = ""
        self.note = ""
        self.stop_requested = False
        self.size_mb = 0.0
        self.size_checked = 0.0


live_session: Optional[LiveSession] = None
_live_lock = threading.Lock()


def read_playlist(sdir: Path):
    """Returns ([(segment_name, duration), ...], total_seconds) from the recording's playlist."""
    try:
        text = (sdir / "index.m3u8").read_text(encoding="utf-8")
    except Exception:
        return [], 0.0
    segs, dur = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            try:
                dur = float(line[8:].split(",")[0])
            except ValueError:
                dur = None
        elif line and not line.startswith("#") and dur is not None:
            segs.append((line, dur))
            dur = None
    return segs, sum(d for _, d in segs)


def kill_proc_tree(p: Optional[subprocess.Popen]):
    if p is None or p.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                           capture_output=True, creationflags=CREATE_NO_WINDOW)
        else:
            p.kill()
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def _pump(stream, tag: str):
    """Forwards a subprocess's stderr into the live console."""
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                state.append_log(f"[LIVE] {tag}: {line}")
    except Exception:
        pass


def _hls_durations(playlist_url: str) -> list:
    import requests
    txt = requests.get(playlist_url, timeout=20).text
    if "#EXT-X-STREAM-INF" in txt:   # master playlist -> read the first variant
        variant = next((l.strip() for l in txt.splitlines() if l.strip() and not l.startswith("#")), None)
        if variant:
            txt = requests.get(urljoin(playlist_url, variant), timeout=20).text
    durs = []
    for line in txt.splitlines():
        if line.startswith("#EXTINF:"):
            try:
                durs.append(float(line[8:].split(",")[0]))
            except ValueError:
                pass
    return durs


def _start_index(durs: list, backfill: str) -> Optional[int]:
    """ffmpeg -live_start_index for the requested amount of history (None = live edge)."""
    if backfill == "now" or not durs:
        return None
    if backfill == "start":
        return 0
    try:
        want = float(backfill)
    except ValueError:
        return None
    acc, idx = 0.0, len(durs)
    while idx > 0 and acc < want:
        idx -= 1
        acc += durs[idx]
    return idx


def _capture_youtube(sess: LiveSession):
    state.append_log(f"[LIVE] Resolving YouTube stream: {sess.url}")
    res = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-g", "--no-warnings", "--no-playlist",
         "-f", "bv*[protocol^=m3u8][height<=1080]+ba[protocol^=m3u8]/b[protocol^=m3u8]/bv*[height<=1080]+ba/best[height<=1080]/best", sess.url],
        capture_output=True, text=True, timeout=60, creationflags=CREATE_NO_WINDOW,
    )
    urls = [u.strip() for u in res.stdout.splitlines() if u.strip().startswith("http")]
    if not urls:
        err = (res.stderr or "").strip().splitlines()
        raise RuntimeError(err[-1] if err else "yt-dlp could not resolve this stream (is it live?)")

    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    reconnect = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]

    if len(urls) >= 2:   # separate video + audio feeds: no history available
        idx = None
        if sess.backfill != "now":
            try:
                durs = _hls_durations(urls[0])
                idx = _start_index(durs, sess.backfill)
                if durs:
                    sess.note = (f"YouTube is exposing {fmt_hms(sum(durs))} of this stream. "
                                 f"Older footage is downloading now and will catch up to live in a moment.")
            except Exception as e:
                state.append_log(f"[LIVE] Could not read stream history ({e}); starting from the live edge.")
        start_opt = ["-live_start_index", str(idx)] if idx is not None else []
        cmd = base + reconnect + start_opt + ["-i", urls[0]] + reconnect + start_opt + ["-i", urls[1]] + \
              ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac"] + HLS_OUT
    else:
        idx = None
        if sess.backfill != "now":
            try:
                durs = _hls_durations(urls[0])
                idx = _start_index(durs, sess.backfill)
                if durs:
                    sess.note = (f"YouTube is exposing {fmt_hms(sum(durs))} of this stream. "
                                 f"Older footage is downloading now and will catch up to live in a moment.")
            except Exception as e:
                state.append_log(f"[LIVE] Could not read stream history ({e}); starting from the live edge.")
        cmd = base + reconnect + (["-live_start_index", str(idx)] if idx is not None else []) + \
              ["-i", urls[0], "-c", "copy"] + HLS_OUT

    state.append_log("[LIVE] Starting recorder...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            cwd=str(sess.dir), creationflags=CREATE_NO_WINDOW)
    sess.procs = [proc]
    threading.Thread(target=_pump, args=(proc.stderr, "ffmpeg"), daemon=True).start()


def _capture_streamlink(sess: LiveSession):
    sess.note = ("Twitch / Kick only keep a short window of earlier footage, so your recording "
                 "starts when you connect.")
    state.append_log(f"[LIVE] Connecting via streamlink: {sess.url}")
    p1 = subprocess.Popen(
        [sys.executable, "-m", "streamlink", "--stdout", "--hls-live-restart", sess.url, "best,720p,worst"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW,
    )
    p2 = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", "pipe:0", "-c", "copy"] + HLS_OUT,
        stdin=p1.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        cwd=str(sess.dir), creationflags=CREATE_NO_WINDOW,
    )
    p1.stdout.close()
    sess.procs = [p1, p2]
    threading.Thread(target=_pump, args=(p1.stderr, "streamlink"), daemon=True).start()
    threading.Thread(target=_pump, args=(p2.stderr, "ffmpeg"), daemon=True).start()


def _run_live_capture(sess: LiveSession):
    try:
        sess.dir.mkdir(parents=True, exist_ok=True)
        if sess.platform == "youtube":
            _capture_youtube(sess)
        else:
            _capture_streamlink(sess)

        ff = sess.procs[-1]
        code = ff.wait()
        if sess.stop_requested:
            return
        _, total = read_playlist(sess.dir)
        if total > 0:
            sess.status = "ended"
            if code != 0:
                sess.error = "The recording stopped unexpectedly. Everything recorded so far is still clippable."
            state.append_log("[LIVE] Recording ended. Everything recorded so far is still clippable.")
        else:
            sess.status = "error"
            sess.error = sess.error or "Could not record this stream. Check that the link is live right now."
            state.append_log(f"[ERROR] {sess.error}")
    except Exception as e:
        if sess.stop_requested:
            return
        sess.status = "error"
        sess.error = str(e)
        state.append_log(f"[ERROR] Live connect failed: {e}")


def _stop_session(s: LiveSession, delete: bool = False):
    s.stop_requested = True
    for p in s.procs:
        kill_proc_tree(p)
    if s.status in ("connecting", "recording"):
        s.status = "stopped"
    if delete:
        for _ in range(4):   # Windows may hold the files for a moment after the kill
            shutil.rmtree(s.dir, ignore_errors=True)
            if not s.dir.exists():
                break
            time.sleep(0.4)


def _shutdown_live():
    s = live_session
    if s and s.status in ("connecting", "recording"):
        _stop_session(s)

atexit.register(_shutdown_live)


@app.post("/api/live/connect")
def live_connect(req: LiveConnectRequest):
    global live_session
    url = req.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Paste the full stream link (starting with https://).")
    backfill = req.backfill.strip().lower()
    if backfill not in ("start", "now") and not backfill.isdigit():
        raise HTTPException(status_code=400, detail="Invalid 'start from' option.")

    with _live_lock:
        cur = live_session
        if cur and cur.status in ("connecting", "recording"):
            raise HTTPException(status_code=400, detail="Already connected to a stream. Disconnect first.")
        sess = LiveSession(url, backfill)
        live_session = sess
    state.append_log(f"[LIVE] New session {sess.id}")
    threading.Thread(target=_run_live_capture, args=(sess,), daemon=True).start()
    return {"status": "connecting", "id": sess.id}


@app.get("/api/live/status")
def live_status():
    s = live_session
    if not s:
        return {"active": False}

    segs, total = read_playlist(s.dir)
    if s.status == "connecting" and total > 0:
        s.status = "recording"
        state.append_log("[LIVE] Recording. You can scrub the timeline and clip now.")

    if s.status in ("connecting", "recording"):
        try:
            if shutil.disk_usage(str(LIVE_DIR)).free < MIN_FREE_BYTES:
                _stop_session(s)
                s.status = "error"
                s.error = "Stopped recording: less than 3 GB of disk space left. Existing footage is still clippable."
                state.append_log(f"[ERROR] {s.error}")
        except Exception:
            pass

    now = time.time()
    if now - s.size_checked > 10:
        try:
            s.size_mb = round(sum(f.stat().st_size for f in s.dir.iterdir() if f.is_file()) / (1024 * 1024), 1)
        except Exception:
            pass
        s.size_checked = now

    return {
        "active": True,
        "id": s.id,
        "status": s.status,
        "error": s.error,
        "note": s.note,
        "platform": s.platform,
        "url": s.url,
        "seconds": round(total, 2),
        "segments": len(segs),
        "size_mb": s.size_mb,
        "playlist": f"/api/live/{s.id}/index.m3u8",
    }


@app.post("/api/live/stop")
def live_stop(req: LiveStopRequest):
    global live_session
    s = live_session
    if not s:
        return {"status": "idle"}
    _stop_session(s, delete=req.delete)
    if req.delete:
        live_session = None
        state.append_log("[LIVE] Disconnected and deleted the recording.")
        return {"status": "deleted"}
    state.append_log("[LIVE] Disconnected. The recording is kept and still clippable.")
    return {"status": "stopped"}


@app.get("/api/live/{sid}/{filename}")
def live_file(sid: str, filename: str):
    """Serves the recording's playlist and segments to the in-app player."""
    if not re.fullmatch(r"[0-9_]{8,20}", sid) or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
        raise HTTPException(status_code=400, detail="Bad request")
    p = LIVE_DIR / sid / filename
    if not p.is_file() or p.suffix not in (".m3u8", ".ts"):
        raise HTTPException(status_code=404, detail="Not found")
    if p.suffix == ".m3u8":
        return FileResponse(str(p), media_type="application/vnd.apple.mpegurl", headers={"Cache-Control": "no-store"})
    return FileResponse(str(p), media_type="video/mp2t", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/live/clip")
async def live_clip(req: LiveClipRequest):
    s = live_session
    if not s:
        raise HTTPException(status_code=400, detail="Connect to a live stream first.")
    if req.layout not in MANUAL_FILTERS:
        raise HTTPException(status_code=400, detail=f"Unknown layout: {req.layout}")

    segs, total = read_playlist(s.dir)
    if not segs:
        raise HTTPException(status_code=400, detail="Nothing has been recorded yet.")

    start = max(0.0, req.start)
    end = min(req.end, total)          # can't clip footage that hasn't been recorded yet
    if end - start < 0.2:
        raise HTTPException(status_code=400, detail="That section isn't recorded yet. Pick an earlier time.")
    if end - start > MAX_MANUAL_CLIP_SECONDS:
        raise HTTPException(status_code=400, detail=f"Clips are limited to {MAX_MANUAL_CLIP_SECONDS // 60} minutes.")

    # Pick only the segments that overlap [start, end]
    chosen, offset, cum = [], 0.0, 0.0
    for name, dur in segs:
        seg_start, seg_end = cum, cum + dur
        if seg_end > start and seg_start < end:
            if not chosen:
                offset = start - seg_start
            chosen.append(name)
        cum = seg_end
    if not chosen:
        raise HTTPException(status_code=400, detail="Could not find footage for that section.")

    if not _manual_lock.acquire(blocking=False):
        raise HTTPException(status_code=400, detail="A manual export is already running. Please wait for it to finish.")

    list_path = s.dir / f"clip_{datetime.now().strftime('%H%M%S%f')}.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for name in chosen:
                f.write(f"file '{(s.dir / name).as_posix()}'\n")

        state.append_log(
            f"[EXPORT] Live clip {fmt_ts(start)} -> {fmt_ts(end)} ({end - start:.1f}s) "
            f"from {len(chosen)} segment(s) [{req.layout}]"
        )
        mreq = ManualClipRequest(file_path="", start=start, end=end, layout=req.layout, name=req.name)
        in_args = ["-f", "concat", "-safe", "0", "-i", str(list_path)]
        out_args = ["-ss", f"{offset:.3f}", "-t", f"{end - start:.3f}"]
        out = await asyncio.to_thread(run_manual_export, mreq, in_args, out_args)
    except Exception as e:
        state.append_log(f"[ERROR] Live clip failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
    finally:
        _manual_lock.release()
        try:
            list_path.unlink()
        except Exception:
            pass

    state.append_log(f"[EXPORT] FINISHED -> manual_exports/{out.name}")
    return {
        "status": "done",
        "filename": out.name,
        "url": f"/api/video/manual_exports/{out.name}",
        "size_mb": round(out.stat().st_size / (1024 * 1024), 2),
    }


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
        cwd=str(PROJECT_ROOT),
        creationflags=CREATE_NO_WINDOW
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
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(state.process.pid)], capture_output=True, creationflags=CREATE_NO_WINDOW)
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

    # 3. Hand-picked clips from manual_exports/
    for p in sorted(MANUAL_EXPORTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True):
        stat = p.stat()
        clips.append({
            "folder": "manual_exports",
            "filename": p.name,
            "title": f"Manual Clip ({p.stem.replace('manual_', '', 1)})",
            "score": None,
            "type": "Manual",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "url": f"/api/video/manual_exports/{p.name}"
        })

    return {"clips": clips}


@app.get("/api/video/{folder}/{filename}")
async def stream_video(folder: str, filename: str):
    bases = {"exports": EXPORTS_DIR, "fallback_exports": FALLBACK_EXPORTS_DIR, "manual_exports": MANUAL_EXPORTS_DIR}
    if folder not in bases:
        raise HTTPException(status_code=400, detail="Invalid folder")
    base = bases[folder]
    target = base / Path(filename).name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path=str(target), media_type="video/mp4", filename=filename)


@app.post("/api/open-folder")
async def open_exports_folder(folder: str = Query("exports")):
    target = {"exports": EXPORTS_DIR, "fallback_exports": FALLBACK_EXPORTS_DIR, "manual_exports": MANUAL_EXPORTS_DIR}.get(folder, EXPORTS_DIR)
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
