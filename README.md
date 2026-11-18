# ClipMore Studio — Autonomous Stream Clipper & Fallback DSP Pipeline

A complete desktop suite for clipping live YouTube, Twitch, and Kick streams into viral 9:16 vertical videos (Shorts, TikTok, Reels) with GPU-accelerated split-screen framing and Faster-Whisper captions.

---

## 🖥️ Native Desktop Application (`ClipMore.exe`)

Simply double-click **`ClipMore.exe`** in `C:\AutoClipper` (or your folder) to launch the native desktop application.

- **No browser needed**: Opens as a standalone Windows desktop app window.
- **No command prompt window**: Clean, background native launcher.
- **Local GPU Accelerated**: Runs entirely on your NVIDIA RTX GPU with zero external cloud dependencies.

*(You can also right-click `ClipMore.exe` → **Show more options** → **Send to** → **Desktop (create shortcut)** to place a shortcut directly on your Windows desktop!)*

---

## 🚀 Web UI Alternative

If you prefer using your standard web browser, double-click **`launch_ui.bat`** or run:
```powershell
.\.venv\Scripts\python.exe app.py
```
And open **`http://127.0.0.1:8501`** in Chrome, Edge, or Brave.

---

## 📂 Workspace Structure

```
C:\AutoClipper\
├── ClipMore.exe                 ← Native desktop application executable
├── main_gui.py                  ← Desktop GUI app controller (pywebview)
├── launch_ui.bat                ← Web browser launcher
├── app.py                       ← Local API server (FastAPI)
│
├── ai\                          ← PRIMARY: AI-Driven Autonomous Clipper
│   ├── auto_viral_clipper.py       YouTube / Twitch / Kick + Whisper CUDA + NVENC
│   ├── run_ai.bat                  Terminal runner
│   └── README.md
│
├── manual\                      ← SECONDARY: Zero-AI Fallback DSP Clipper
│   ├── fallback_clipper.py         Acoustic volume peak thresholding
│   ├── run_fallback.bat            Terminal runner
│   └── README.md
│
├── web\                         ← App UI templates & styling
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

### 1. AI Autonomous Live Clipper (YouTube / Twitch / Kick)
```powershell
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py "https://www.youtube.com/watch?v=..."
```

### 2. Fallback DSP Clipper (Local Video / VOD)
```powershell
.\.venv\Scripts\python.exe manual\fallback_clipper.py path\to\video.mp4 --max-clips 5
```

### 3. Pipeline Self-Test
```powershell
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py --test
```
