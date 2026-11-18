# Fallback Clipper — Zero AI Tokens, Pure Offline DSP

Deterministic audio-energy clipper. No neural networks, no chat analysis, no cloud dependencies. Works on pre-recorded VODs and local video files.

## How It Works
- Extracts mono 16kHz audio → computes 2-second RMS energy windows
- Identifies spikes > 2σ above mean (screaming, laughter, loud reactions)
- Renders each spike as a 9:16 vertical split-screen clip via NVENC

## Usage
```powershell
cd C:\AutoClipper
.\.venv\Scripts\python.exe manual\fallback_clipper.py path\to\video.mp4
.\.venv\Scripts\python.exe manual\fallback_clipper.py path\to\vod.mp4 --max-clips 10
```

## Self-Test
```powershell
.\.venv\Scripts\python.exe manual\fallback_clipper.py --test
```
