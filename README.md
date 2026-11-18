# ClipMore Studio — Autonomous Stream Clipper & Fallback DSP Pipeline

A complete desktop suite for clipping live Twitch, YouTube, and Kick streams into viral 9:16 vertical videos (Shorts, TikTok, Reels) with GPU-accelerated split-screen framing and Faster-Whisper captions.

---

## 🚀 Easy Web UI (Recommended)

Double-click **`launch_ui.bat`** or run:
```powershell
.\.venv\Scripts\python.exe app.py
```
Open **`http://127.0.0.1:8501`** in any browser.

### Web UI Features
- **One-Click Live Clipping**: Paste any Twitch / Kick / YouTube URL and click Start.
- **GPU Dashboard**: Live NVIDIA GeForce RTX GPU utilization, VRAM, and temperature monitoring.
- **Real-Time Terminal**: Watch live chat velocity, audio RMS spikes, and Whisper captions stream in the browser.
- **Built-in Mobile Player**: Preview finished 9:16 vertical clips with audio right inside the dashboard.
- **Direct Downloads**: Download clips or click "Open Exports" to view them in Windows File Explorer.
- **Mode Toggle**: Switch seamlessly between **AI Autonomous Live Clipper** and **Zero-AI Fallback DSP Clipper**.

---

## 📂 Workspace Structure

```
C:\AutoClipper\ (c:\Users\A S U S\Documents\Clipping)
├── launch_ui.bat                ← Double-click to start Web UI
├── app.py                       ← Web UI backend server (FastAPI)
│
├── ai\                          ← PRIMARY: AI-Driven Autonomous Clipper
│   ├── auto_viral_clipper.py       Live Chat Velocity + Audio RMS + Whisper CUDA + NVENC
│   ├── run_ai.bat                  Double-click CLI launcher
│   └── README.md
│
├── manual\                      ← SECONDARY: Zero-AI Fallback DSP Clipper
│   ├── fallback_clipper.py         Pure acoustic energy thresholding (>2σ)
│   ├── run_fallback.bat            Double-click CLI launcher
│   └── README.md
│
├── web\                         ← Web UI Templates & UI assets
│   └── templates/index.html
│
├── exports\                     ← Finished 9:16 vertical clips (AI mode)
├── fallback_exports\            ← Finished 9:16 vertical clips (Fallback mode)
├── raw_chunks\                  ← Rolling .ts stream buffer
├── .agents/skills/viral-stream-clipper/
│   └── SKILL.md                 ← Antigravity Agent Skill
└── .venv\                       ← Python 3.12 + PyTorch/CUDA environment
```

---

## 💻 Command Line Execution

### 1. AI Autonomous Clipper (Live Stream)
```powershell
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py "https://www.twitch.tv/channelname"
```

### 2. Fallback DSP Clipper (Local Video / VOD)
```powershell
.\.venv\Scripts\python.exe manual\fallback_clipper.py path\to\video.mp4 --max-clips 5
```

### 3. Self-Test
```powershell
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py --test
.\.venv\Scripts\python.exe manual\fallback_clipper.py --test
```
