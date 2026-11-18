# AutoClipper — Two Independent Clipping Systems

## Workspace Layout

```
C:\AutoClipper\
├── ai\                         ← PRIMARY: AI-Driven Autonomous Clipper
│   ├── auto_viral_clipper.py      Neural Whisper + Chat Velocity + Audio RMS
│   ├── run_ai.bat                 Double-click launcher
│   └── README.md
│
├── manual\                     ← SECONDARY: Zero-AI Fallback Clipper
│   ├── fallback_clipper.py        Pure DSP audio energy spike detection
│   ├── run_fallback.bat           Double-click launcher
│   └── README.md
│
├── .agents/skills/viral-stream-clipper/
│   └── SKILL.md                   Antigravity Agent Skill (auto-triggers on stream links)
│
├── .venv\                      ← Shared Python 3.12 virtual environment
├── raw_chunks\                 ← Rolling .ts stream buffer (AI mode)
├── exports\                    ← Finished 9:16 clips (AI mode)
├── fallback_exports\           ← Finished 9:16 clips (Fallback mode)
└── temp\                       ← Intermediate files
```

---

## System 1: AI Autonomous Clipper (`ai/`)

| Feature | Detail |
|:---|:---|
| Detection | Chat velocity spikes + Audio RMS peaks + Whisper transcript hooks |
| Transcription | faster-whisper `small.en` on CUDA float16 |
| Captions | Auto-generated `.ass` (3 uppercase words/line, TikTok style) |
| Rendering | 9:16 split-screen (facecam top, gameplay bottom) via h264_nvenc |
| Input | Live Twitch / YouTube / Kick stream URL |

```powershell
cd C:\AutoClipper
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py "https://www.twitch.tv/channelname"
```

---

## System 2: Fallback DSP Clipper (`manual/`)

| Feature | Detail |
|:---|:---|
| Detection | Pure audio RMS energy thresholding (>2σ) |
| AI Required | None — zero tokens, zero neural networks |
| Captions | None (raw video export) |
| Rendering | 9:16 split-screen via h264_nvenc |
| Input | Pre-recorded VOD or local video file |

```powershell
cd C:\AutoClipper
.\.venv\Scripts\python.exe manual\fallback_clipper.py path\to\video.mp4
```
