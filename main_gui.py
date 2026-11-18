#!/usr/bin/env python3
"""
ClipMore Studio - Native Desktop Application Entry Point
-------------------------------------------------------
Runs the local backend in the background and opens a native Windows
desktop application window using pywebview (Edge WebView2 runtime).
"""

import os
import sys
import time
import socket
import threading
from pathlib import Path

# Ensure working directory is project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

def find_free_port(start_port=8501):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port

def run_server(port):
    import uvicorn
    from app import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

def main():
    import webview
    
    port = find_free_port(8501)
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait for local server port to become reachable
    for _ in range(50):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    break
        except Exception:
            pass
        time.sleep(0.1)

    url = f"http://127.0.0.1:{port}"
    
    window = webview.create_window(
        title="ClipMore Studio - Autonomous Stream Clipper",
        url=url,
        width=1280,
        height=850,
        min_size=(960, 620),
        background_color="#0b0f19"
    )
    
    webview.start(debug=False)

if __name__ == "__main__":
    main()
