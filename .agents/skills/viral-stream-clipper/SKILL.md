---
name: viral-stream-clipper
description: Autonomously find, score, and export multiple viral 9:16 vertical clips from Twitch, YouTube, or Kick streams using live chat velocity, audio RMS spikes, Whisper captions, and NVENC GPU split-screen rendering. Trigger whenever a user pastes a stream link and asks to clip it.
---

# Viral Stream Clipper Skill

Autonomous live-stream viral clipper with two operational modes.

## When to Activate
Activate automatically whenever the user pastes a Twitch, YouTube, or Kick stream URL and asks to clip, highlight, or create TikToks/Shorts/Reels from it.

## Execution Steps

### 1. Launch the AI Clipper in Background
```powershell
cd C:\AutoClipper
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py "<STREAM_URL>"
```
This starts two threads:
- **Thread A**: chat-downloader tracks live chat velocity & hype keywords
- **Thread B**: streamlink → ffmpeg segments → audio RMS → faster-whisper CUDA → viral score → 9:16 NVENC render

### 2. Auto-Calibrate Webcam Crop
After 15s of recording, extract a frame from the latest chunk to verify webcam position:
```powershell
ffmpeg -y -sseof -2 -i (Get-ChildItem raw_chunks/*.ts | Select-Object -Last 1).FullName -vframes 1 temp/calibration_frame.jpg
```

### 3. Monitor and Report Clips
Watch `exports/` for new clips. Report each with its filename (which encodes timestamp + score + hook).

### 4. Fallback Mode
If AI tokens are exhausted or Whisper is unavailable, switch to the DSP-only fallback:
```powershell
.\.venv\Scripts\python.exe manual\fallback_clipper.py path\to\video.mp4
```
