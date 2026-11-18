#!/usr/bin/env python3
"""
PRIMARY AI-DRIVEN Autonomous Live Stream Viral Clipper
======================================================
Multi-Platform Support: YouTube, Twitch, Kick, and VODs

Architecture:
  [Live Stream URL (Twitch / YouTube / Kick)]
         │
         ├──► (Thread A) Chat Velocity & Hype Spikes (chat-downloader / Twitch IRC)
         │
         └──► (Thread B) Stream Capture (yt-dlp for YouTube, streamlink for Twitch)
                                  │
                                  ▼
                         FFmpeg Segmenter (.ts chunks in raw_chunks/)
                                  │
                                  ▼
                       Audio Energy (RMS) & VAD Filter
                                  │
                                  ▼
                     faster-whisper (CUDA float16)
                                  │
                                  ▼
                     Viral Ranking & .ass Generation
                                  │
                                  ▼
                     FFmpeg GPU (h264_nvenc) Split-Screen Render
"""

import os
import sys
import time
import re
import socket
import threading
import subprocess
import numpy as np
from pathlib import Path
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════
#  CUDA DLL Bootstrap — must run BEFORE importing faster_whisper / ctranslate2
# ═══════════════════════════════════════════════════════════════════════════
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def _bootstrap_cuda_and_ffmpeg():
    """Add NVIDIA cuBLAS/cuDNN DLL dirs to PATH so CTranslate2 can find them,
    and ensure FFmpeg is also on PATH."""
    venv_sp = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "nvidia"
    if venv_sp.exists():
        for sub in venv_sp.iterdir():
            bin_dir = sub / "bin"
            if bin_dir.is_dir() and str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(bin_dir))
                    except OSError:
                        pass

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
#  THREAD A — Live Chat Velocity & Keyword Spike Monitor (Twitch + YouTube + Kick)
# ═══════════════════════════════════════════════════════════════════════════
def monitor_chat(stream_url: str):
    """Continuously monitors live chat velocity and keyword spikes for Twitch, YouTube, or Kick."""
    is_youtube = "youtube.com" in stream_url.lower() or "youtu.be" in stream_url.lower()
    is_twitch = "twitch.tv" in stream_url.lower()

    if is_youtube:
        log("CHAT", f"Starting YouTube live chat monitor for: {stream_url}")
        _run_chat_downloader(stream_url, platform_name="YouTube")
    elif is_twitch:
        channel_name = stream_url.lower().rstrip("/").split("/")[-1]
        log("CHAT", f"Starting Twitch chat monitor for channel #{channel_name}")
        _run_twitch_irc(channel_name)
    else:
        log("CHAT", f"Starting universal chat-downloader for: {stream_url}")
        _run_chat_downloader(stream_url, platform_name="Universal")


def _run_twitch_irc(channel_name: str):
    """Anonymous Twitch IRC connection with 0 API tokens."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10.0)
        s.connect(("irc.chat.twitch.tv", 6667))
        nick = f"justinfan{int(time.time()) % 100000}"
        s.sendall(f"PASS SCHMOOPIE\r\nNICK {nick}\r\nJOIN #{channel_name}\r\n".encode("utf-8"))
        log("CHAT", f"Connected to Twitch chat as {nick}.")
        buf = ""
        window = []

        while True:
            try:
                data = s.recv(4096).decode("utf-8", errors="ignore")
            except socket.timeout:
                data = ""

            if data:
                buf += data
                lines = buf.split("\r\n")
                buf = lines.pop()
                now = time.time()
                for line in lines:
                    if line.startswith("PING"):
                        s.sendall("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
                    elif "PRIVMSG" in line:
                        parts = line.split(f"#{channel_name} :", 1)
                        if len(parts) > 1:
                            _process_chat_message(now, parts[1], window)
    except Exception as e:
        log("CHAT", f"Twitch chat notice: {e}")


def _run_chat_downloader(stream_url: str, platform_name: str = "Platform"):
    """Uses chat-downloader to parse YouTube Live, Kick, or other stream chats."""
    try:
        from chat_downloader import ChatDownloader
        chat = ChatDownloader().get_chat(stream_url)
        window = []
        log("CHAT", f"Connected to {platform_name} live chat stream.")
        for message in chat:
            now = time.time()
            text = message.get("message", "")
            _process_chat_message(now, text, window)
    except Exception as e:
        log("CHAT", f"{platform_name} chat notice: {e}")
        log("CHAT", "Notice: If live chat is unavailable, Audio RMS Energy & Whisper speech scoring will continue autonomously!")


def _process_chat_message(now: float, message_text: str, window: list):
    window.append(now)
    # 10-second rolling window
    cutoff = now - 10.0
    while window and window[0] < cutoff:
        window.pop(0)

    upper_text = message_text.upper()
    triggers = ["CLIP", "W", "LUL", "OMG", "NO WAY", "INSANE", "WHAT", "SHEESH",
                "CLUTCH", "POG", "POGGERS", "KEKW", "HOLY", "WTF", "YOOO", "LOL"]
    has_trigger = any(trg in upper_text for trg in triggers)

    ts_bucket = int(now // 10) * 10
    chat_hype_scores[ts_bucket] = chat_hype_scores.get(ts_bucket, 0) + (3 if has_trigger else 1)

    if len(window) > 0 and int(now) % 30 == 0:
        rate = len(window) / 10.0
        log("CHAT", f"Chat Velocity: {rate:.1f} msg/s | Spike Buckets Tracked: {len(chat_hype_scores)}")


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD B — Multi-Platform Stream Capture (YouTube / Twitch / Kick)
# ═══════════════════════════════════════════════════════════════════════════
def start_capture_process(stream_url: str):
    """
    Starts stream ingestion into raw_chunks/chunk_%04d.ts.
    Automatically chooses yt-dlp for YouTube, and streamlink for Twitch/Kick.
    """
    is_youtube = "youtube.com" in stream_url.lower() or "youtu.be" in stream_url.lower()
    chunk_pattern = str(RAW_CHUNKS / "chunk_%04d.ts")

    if is_youtube:
        log("CAPTURE", f"Resolving YouTube stream via yt-dlp for: {stream_url}")
        try:
            # First try direct HLS URL resolution
            cmd_url = [sys.executable, "-m", "yt_dlp", "-g", "--no-warnings", "-f", "best[ext=mp4]/best", stream_url]
            res = subprocess.run(cmd_url, capture_output=True, text=True, timeout=15)
            direct_hls = res.stdout.strip().split("\n")[0]
            if res.returncode == 0 and direct_hls.startswith("http"):
                log("CAPTURE", "Resolved direct YouTube stream URL. Connecting FFmpeg segmenter...")
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                    "-reconnect", "1", "-reconnect_at_eof", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                    "-i", direct_hls,
                    "-c", "copy", "-f", "segment", "-segment_time", "180", "-reset_timestamps", "1",
                    chunk_pattern
                ]
                return subprocess.Popen(ffmpeg_cmd)
        except Exception as e:
            log("CAPTURE", f"YouTube direct HLS notice: {e}, falling back to piped yt-dlp...")

        # Pipe fallback with yt-dlp
        log("CAPTURE", "Piping YouTube live stream through yt-dlp -> FFmpeg...")
        p1 = subprocess.Popen(
            [sys.executable, "-m", "yt_dlp", "-o", "-", "-f", "best[height<=1080]/best", "--no-warnings", stream_url],
            stdout=subprocess.PIPE
        )
        p2 = subprocess.Popen(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", "pipe:0",
             "-c", "copy", "-f", "segment", "-segment_time", "180", "-reset_timestamps", "1", chunk_pattern],
            stdin=p1.stdout
        )
        p1.stdout.close()
        return p2

    else:
        # Twitch / Kick: streamlink pipe
        log("CAPTURE", f"Piping live stream via streamlink -> FFmpeg for: {stream_url}...")
        p1 = subprocess.Popen(
            [sys.executable, "-m", "streamlink", "--stdout", stream_url, "best,720p,worst"],
            stdout=subprocess.PIPE
        )
        p2 = subprocess.Popen(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", "pipe:0",
             "-c", "copy", "-f", "segment", "-segment_time", "180", "-reset_timestamps", "1", chunk_pattern],
            stdin=p1.stdout
        )
        p1.stdout.close()
        return p2


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
    # Top 1/3: Crop webcam region (top-left/creator box) → scale to 1080x640
    # Bottom 2/3: Center crop gameplay → scale to 1080x1280
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
        # Fallback to libx264 if NVENC reports error
        cmd[cmd.index("h264_nvenc")] = "libx264"
        cmd[cmd.index("p4")] = "fast"
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ═══════════════════════════════════════════════════════════════════════════
#  Chunk Processor (Audio RMS Spikes + Whisper CUDA + Viral Render)
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
        return

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
        log("SCORER", f"{base_name} → no high-energy peaks, skipping.")
        return

    log("AI", f"Found {len(peaks)} audio reaction peaks in {base_name}. Transcribing speech on CUDA...")

    for idx, peak_sec in enumerate(peaks):
        start_sec = max(0, peak_sec - 20)
        duration = 35

        # Transcribe candidate window via numpy
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
        segments = list(segments)

        transcript_text = " ".join([seg.text for seg in segments])
        if len(transcript_text.strip()) < 10:
            continue

        log("WHISPER", f"Transcript: \"{transcript_text.strip()[:80]}...\"")

        ass_path = str(EXPORTS / f"{base_name}_clip{idx}.ass")
        out_mp4  = str(EXPORTS / f"{base_name}_clip{idx}.mp4")

        generate_ass_file(segments, 0.0, ass_path)
        render_vertical_clip(ts_path, start_sec, duration, ass_path, out_mp4)

        if os.path.exists(ass_path):
            os.remove(ass_path)
        log("EXPORT", f"🌟 FINISHED VIRAL CLIP: {out_mp4}")


# ═══════════════════════════════════════════════════════════════════════════
#  Self-Test Mode
# ═══════════════════════════════════════════════════════════════════════════
def _run_test():
    log("TEST", "Running multi-platform AI pipeline self-test...")

    test_src = str(EXPORTS / "ai_test_src.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-pix_fmt", "yuv420p", "-shortest", test_src
    ], check=True)

    test_wav = str(EXPORTS / "ai_test_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", test_src, "-ac", "1", "-ar", "16000", test_wav
    ], check=True)

    sr, data = wavfile.read(test_wav)
    audio_np = data.astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(audio_np, word_timestamps=True)
    segments = list(segments)
    log("TEST", f"Whisper returned {len(segments)} segments on CUDA ✅")

    ass_path = str(EXPORTS / "ai_test.ass")
    out_mp4  = str(EXPORTS / "ai_test_clip.mp4")

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

    for f in [test_src, test_wav, ass_path]:
        if os.path.exists(f):
            os.remove(f)

    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000:
        log("TEST", f"🎉 Multi-platform AI pipeline PASSED → {out_mp4}")
    else:
        log("TEST", "❌ Test FAILED")


# ═══════════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_test()
        sys.exit(0)

    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter Stream URL (YouTube, Twitch, or Kick): ")

    # Thread A: Live Chat Velocity Monitor
    threading.Thread(target=monitor_chat, args=(url,), daemon=True).start()

    # Thread B: Stream Ingestion via yt-dlp (YouTube) or streamlink (Twitch)
    log("SYSTEM", f"Starting multi-platform stream capture for: {url}")
    capture_proc = start_capture_process(url)

    log("SYSTEM", "Thread A (Chat Velocity) + Thread B (Capture & Scorer) are running.")
    log("SYSTEM", f"Finished clips will appear in: {EXPORTS}")

    processed = set()
    try:
        while True:
            files = sorted(f for f in os.listdir(RAW_CHUNKS) if f.endswith(".ts"))
            for f in files[:-1]:
                full_path = os.path.join(str(RAW_CHUNKS), f)
                if full_path not in processed:
                    process_ts_chunk(full_path)
                    processed.add(full_path)
            time.sleep(10)
    except KeyboardInterrupt:
        log("SYSTEM", "Stopping clipper...")
        if capture_proc:
            capture_proc.terminate()
