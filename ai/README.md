# AI Autonomous Viral Stream Clipper

Fully autonomous — paste a live stream URL and walk away. Uses local GPU compute (RTX NVENC + CUDA Whisper).

## How It Works
- **Thread A**: `chat-downloader` monitors live chat velocity & keyword spikes in real-time
- **Thread B**: `streamlink --stdout` → FFmpeg segments rolling `.ts` chunks → Audio RMS peaks → `faster-whisper` CUDA transcription → Viral scoring → 9:16 split-screen NVENC render

## Usage
```powershell
cd C:\AutoClipper
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py "https://www.twitch.tv/channelname"
```

## Self-Test
```powershell
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py --test
```
