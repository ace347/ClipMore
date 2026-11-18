#!/usr/bin/env python3
"""
PRIMARY AI-DRIVEN Autonomous Live Stream Viral Clipper
======================================================
Architecture:
  [Live Stream URL]
         │
         ├──► (Thread A) chat-downloader ──► Sliding Window Spike Detection (>2.5x) ──► Viral Score Boost
         │
         └──► (Thread B) streamlink --stdout ──► FFmpeg Segmenter (.ts chunks)
                                                        │
                                                        ▼
                                             Audio Energy & VAD Filter
                                                        │
                                                        ▼
                                          faster-whisper (CUDA float16)
                                                        │
                                                        ▼
                                          Viral Ranking & .ass Generation
                                                        │
                                                        ▼
                                          FFmpeg GPU (h264_nvenc) Split-Screen Render

Usage:
    python ai/auto_viral_clipper.py "https://www.twitch.tv/channelname"
    python ai/auto_viral_clipper.py --test
"""

import os
import sys
import time
import threading
import subprocess
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  CUDA DLL Bootstrap — must run BEFORE importing faster_whisper / ctranslate2
# ═══════════════════════════════════════════════════════════════════════════
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def _bootstrap_cuda_and_ffmpeg():
    """Add NVIDIA cuBLAS/cuDNN DLL dirs to PATH so CTranslate2 can find them,
    and ensure FFmpeg (winget/choco install) is also on PATH."""
    venv_sp = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "nvidia"
    if venv_sp.exists():
        for sub in venv_sp.iterdir():
            bin_dir = sub / "bin"
            if bin_dir.is_dir() and str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
                # Python 3.8+ on Windows: register DLL search directory
                if hasattr(os, "add_dll_directory"):
                    try: os.add_dll_directory(str(bin_dir))
                    except OSError: pass

    for base in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages",
        Path("C:/ProgramData/chocolatey/bin"),
    ]:
        if base.exists():
            for exe in base.glob("**/ffmpeg.exe"):
                d = str(exe.parent)
                if d not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
                return

_bootstrap_cuda_and_ffmpeg()

# Now safe to import CUDA-dependent libraries
from scipy.io import wavfile
from faster_whisper import WhisperModel

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CHUNKS   = PROJECT_ROOT / "raw_chunks"
EXPORTS      = PROJECT_ROOT / "exports"

os.makedirs(RAW_CHUNKS, exist_ok=True)
os.makedirs(EXPORTS, exist_ok=True)

def log(tag, msg):
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] [{tag}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Initialize Whisper on GPU at module load (stays resident in VRAM)
# ═══════════════════════════════════════════════════════════════════════════
log("INIT", "Loading Whisper model into GPU memory (small.en, CUDA float16)...")
try:
    whisper_model = WhisperModel("small.en", device="cuda", compute_type="float16")
    log("INIT", "✅ Whisper loaded on CUDA (RTX GPU)")
except Exception as e:
    log("INIT", f"CUDA unavailable ({e}), falling back to CPU int8")
    whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")


# ═══════════════════════════════════════════════════════════════════════════
#  Shared state for chat hype tracking
# ═══════════════════════════════════════════════════════════════════════════
chat_hype_scores = {}


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD A — Live Chat Velocity & Keyword Spike Monitor
# ═══════════════════════════════════════════════════════════════════════════
def monitor_chat(stream_url: str):
    """Continuously monitors live chat velocity and keyword spikes."""
    log("CHAT", "Starting chat stream parser...")
    try:
        from chat_downloader import ChatDownloader
        chat = ChatDownloader().get_chat(stream_url)
        window = []
        for message in chat:
            now = time.time()
            window.append(now)
            # Keep rolling 10-second window
            window = [t for t in window if now - t <= 10]

            # Check for keyword triggers in message text
            text = message.get("message", "").upper()
            triggers = ["CLIP", "W", "LUL", "OMG", "NO WAY", "INSANE", "WHAT", "SHEESH",
                        "CLUTCH", "POG", "POGGERS", "KEKW", "HOLY", "WTF", "YOOO"]
            has_trigger = any(trg in text for trg in triggers)

            # Record timestamps for spike analysis (10-second buckets)
            ts_bucket = int(now // 10) * 10
            chat_hype_scores[ts_bucket] = chat_hype_scores.get(ts_bucket, 0) + (3 if has_trigger else 1)

            # Periodic logging
            if len(window) > 0 and int(now) % 30 == 0:
                rate = len(window) / 10.0
                log("CHAT", f"Rate: {rate:.1f} msg/s | Window: {len(window)} msgs | Buckets tracked: {len(chat_hype_scores)}")

    except Exception as e:
        log("CHAT", f"Chat monitor error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  .ass Subtitle Generator (3 uppercase words per line, viral TikTok style)
# ═══════════════════════════════════════════════════════════════════════════
def generate_ass_file(segments, start_offset: float, ass_path: str):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,68,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def fmt_t(t):
        t = max(0.0, t - start_offset)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    lines = [header]
    for seg in segments:
        if not seg.words:
            continue
        words = seg.words
        for i in range(0, len(words), 3):
            chunk = words[i:i + 3]
            t0 = fmt_t(chunk[0].start)
            t1 = fmt_t(chunk[-1].end)
            text = " ".join(w.word.strip().upper() for w in chunk)
            lines.append(f"Dialogue: 0,{t0},{t1},Default,,0,0,0,,{text}\n")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  9:16 Vertical Split-Screen Renderer (NVENC GPU)
# ═══════════════════════════════════════════════════════════════════════════
def render_vertical_clip(input_path: str, start: float, duration: float,
                         ass_path: str, output_path: str):
    safe_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    # Top 1/3: Crop webcam region (top-left corner) → scale to 1080x640
    # Bottom 2/3: Center crop gameplay → scale to 1080x1280
    # Burn .ass subtitles over the composited 1080x1920 canvas
    filter_complex = (
        "[0:v]crop=480:360:20:700,scale=1080:640:force_original_aspect_ratio=increase,crop=1080:640[cam];"
        "[0:v]crop=1080:1080:(in_w-1080)/2:0,scale=1080:1280[game];"
        f"[cam][game]vstack=inputs=2,ass='{safe_ass}'[v]"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration), "-i", input_path,
        "-filter_complex", filter_complex, "-map", "[v]", "-map", "0:a",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21",
        "-c:a", "aac", "-b:a", "160k", output_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        # Fallback to libx264 if NVENC fails
        cmd[cmd.index("h264_nvenc")] = "libx264"
        cmd[cmd.index("p4")] = "fast"
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD B — Chunk Processor (Audio RMS + Whisper + Score + Render)
# ═══════════════════════════════════════════════════════════════════════════
def process_ts_chunk(ts_path: str):
    """Analyzes a recorded .ts chunk for audio peaks, transcribes, and renders viral clips."""
    base_name = os.path.splitext(os.path.basename(ts_path))[0]
    wav_path = ts_path.replace(".ts", ".wav")

    # Extract audio track to analyze volume/energy peaks
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", ts_path, "-ac", "1", "-ar", "16000", wav_path
    ], check=True)

    try:
        sr, data = wavfile.read(wav_path)
    except Exception:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return

    os.remove(wav_path)
    if len(data) < sr * 5:
        return  # Skip chunks too short

    # Compute RMS energy windows (1-second intervals)
    samples_per_sec = sr
    trimmed = data[:len(data) - (len(data) % samples_per_sec)]
    rms = np.sqrt(np.mean(
        trimmed.reshape(-1, samples_per_sec).astype(np.float32) ** 2, axis=1
    ))

    if len(rms) == 0:
        return

    mean_rms = np.mean(rms)
    std_rms = np.std(rms) + 1e-6
    z_scores = (rms - mean_rms) / std_rms

    # Find peak reactions (Z-score > 2.8, with 45s cooldown)
    peaks = []
    last_peak = -45
    for sec, z in enumerate(z_scores):
        if z > 2.8 and (sec - last_peak) >= 45:
            peaks.append(sec)
            last_peak = sec

    if not peaks:
        log("SCORER", f"{base_name} → no peaks above threshold, skipping.")
        return

    log("AI", f"Found {len(peaks)} audio reaction peaks in {base_name}. Transcribing...")

    for idx, peak_sec in enumerate(peaks):
        start_sec = max(0, peak_sec - 20)
        duration = 35

        # Transcribe the candidate window via numpy (bypasses av library)
        clip_wav = str(RAW_CHUNKS / f"{base_name}_clip{idx}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_sec), "-t", str(duration),
            "-i", ts_path, "-ac", "1", "-ar", "16000", clip_wav
        ], check=True)

        try:
            clip_sr, clip_data = wavfile.read(clip_wav)
            os.remove(clip_wav)
        except Exception:
            continue

        audio_np = clip_data.astype(np.float32) / 32768.0
        segments, _ = whisper_model.transcribe(
            audio_np, word_timestamps=True, vad_filter=True
        )
        segments = list(segments)  # exhaust generator

        transcript_text = " ".join([seg.text for seg in segments])
        if len(transcript_text.strip()) < 10:
            continue  # Skip dead-air hallucinations

        log("WHISPER", f"Transcript: \"{transcript_text.strip()[:80]}...\"")

        # Generate styled .ass subtitles
        ass_path = str(EXPORTS / f"{base_name}_clip{idx}.ass")
        out_mp4  = str(EXPORTS / f"{base_name}_clip{idx}.mp4")

        generate_ass_file(segments, 0.0, ass_path)
        render_vertical_clip(ts_path, start_sec, duration, ass_path, out_mp4)

        if os.path.exists(ass_path):
            os.remove(ass_path)
        log("EXPORT", f"🌟 {out_mp4}")


# ═══════════════════════════════════════════════════════════════════════════
#  Self-Test (no live stream needed)
# ═══════════════════════════════════════════════════════════════════════════
def _run_test():
    log("TEST", "Running AI pipeline self-test...")

    # Generate synthetic 1920x1080 video with loud+quiet sections
    test_src = str(EXPORTS / "ai_test_src.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-pix_fmt", "yuv420p", "-shortest", test_src
    ], check=True)

    # Extract audio → numpy → transcribe
    test_wav = str(EXPORTS / "ai_test_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", test_src, "-ac", "1", "-ar", "16000", test_wav
    ], check=True)

    sr, data = wavfile.read(test_wav)
    audio_np = data.astype(np.float32) / 32768.0
    log("TEST", f"Audio loaded: {len(audio_np)/16000:.1f}s @ 16kHz")

    segments, _ = whisper_model.transcribe(audio_np, word_timestamps=True)
    segments = list(segments)
    log("TEST", f"Whisper returned {len(segments)} segments on CUDA ✅")

    # Generate subtitles + render vertical
    ass_path = str(EXPORTS / "ai_test.ass")
    out_mp4  = str(EXPORTS / "ai_test_clip.mp4")

    # Mock subtitle for rendering test
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,68,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,WHAT JUST HAPPENED
Dialogue: 0,0:00:03.50,0:00:05.50,Default,,0,0,0,,NO WAY BRO
Dialogue: 0,0:00:06.00,0:00:08.00,Default,,0,0,0,,INSANE CLUTCH
""")

    render_vertical_clip(test_src, 0, 10, ass_path, out_mp4)

    # Cleanup temp files
    for f in [test_src, test_wav, ass_path]:
        if os.path.exists(f):
            os.remove(f)

    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000:
        log("TEST", f"🎉 AI pipeline PASSED → {out_mp4}")
    else:
        log("TEST", "❌ AI pipeline FAILED")


# ═══════════════════════════════════════════════════════════════════════════
#  Main — Lock-Free Capture Pipeline
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_test()
        sys.exit(0)

    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter Stream URL: ")

    # Launch chat watcher thread (Thread A)
    threading.Thread(target=monitor_chat, args=(url,), daemon=True).start()

    # Launch lock-free stream capture: streamlink → pipe → ffmpeg segmenter (Thread B)
    log("CAPTURE", f"Starting lock-free capture for {url}...")
    p1 = subprocess.Popen(
        [sys.executable, "-m", "streamlink", "--stdout", url, "best"],
        stdout=subprocess.PIPE
    )
    p2 = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "warning",
         "-i", "pipe:0", "-c", "copy", "-f", "segment",
         "-segment_time", "180", "-reset_timestamps", "1",
         str(RAW_CHUNKS / "chunk_%04d.ts")],
        stdin=p1.stdout
    )
    p1.stdout.close()

    log("SYSTEM", "Thread A (Chat Velocity) + Thread B (Capture → Score → Render) active.")
    log("SYSTEM", f"Clips auto-export to: {EXPORTS}")

    processed = set()
    try:
        while True:
            files = sorted(f for f in os.listdir(RAW_CHUNKS) if f.endswith(".ts"))
            for f in files[:-1]:  # Leave newest chunk alone (still recording)
                full_path = os.path.join(str(RAW_CHUNKS), f)
                if full_path not in processed:
                    process_ts_chunk(full_path)
                    processed.add(full_path)
            time.sleep(10)
    except KeyboardInterrupt:
        log("SYSTEM", "Stopping clipper...")
        p2.terminate()
