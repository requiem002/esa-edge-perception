#!/usr/bin/env python3
"""
Captures the MuJoCo viewer window and serves it as MJPEG on port 8093.
Uses ffmpeg x11grab + xdotool for window detection.

Run: python3 sim_capture.py [--port 8093] [--fps 10] [--display :0]
"""

import argparse
import asyncio
import os
import subprocess
import time

os.environ.setdefault("DISPLAY", ":0")

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

# ─── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="MuJoCo window MJPEG capture")
parser.add_argument("--port", type=int, default=8093)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--display", default=":0")
args = parser.parse_args()

WINDOW_NAMES = ["MuJoCo", "unitree", "Go2", "Fenrir", "mujoco"]

app = FastAPI()
_ffmpeg_proc = None
_window_geom = None   # {"x": int, "y": int, "w": int, "h": int}
_last_check = 0.0

# ─── Window detection ─────────────────────────────────────────────────────────

def find_window_geom():
    """Use xdotool to find MuJoCo window geometry. Returns dict or None."""
    for name in WINDOW_NAMES:
        try:
            result = subprocess.check_output(
                ["xdotool", "search", "--name", name],
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue

        if not result:
            continue

        wid = result.splitlines()[0].strip()
        try:
            geo = subprocess.check_output(
                ["xdotool", "getwindowgeometry", "--shell", wid],
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).decode()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

        geom = {}
        for line in geo.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                geom[k.strip()] = v.strip()

        try:
            return {
                "x": int(geom.get("X", 0)),
                "y": int(geom.get("Y", 0)),
                "w": int(geom.get("WIDTH", 1280)),
                "h": int(geom.get("HEIGHT", 720)),
            }
        except (ValueError, KeyError):
            continue

    return None

# ─── ffmpeg process ───────────────────────────────────────────────────────────

def start_ffmpeg(geom):
    global _ffmpeg_proc
    if _ffmpeg_proc and _ffmpeg_proc.poll() is None:
        _ffmpeg_proc.terminate()
        try:
            _ffmpeg_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _ffmpeg_proc.kill()
        _ffmpeg_proc = None

    x, y, w, h = geom["x"], geom["y"], geom["w"], geom["h"]
    display = args.display
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-f", "x11grab",
        "-r", str(args.fps),
        "-s", f"{w}x{h}",
        "-i", f"{display}+{x},{y}",
        "-vcodec", "mjpeg",
        "-q:v", "4",
        "-f", "mpjpeg",
        "pipe:1",
    ]
    try:
        _ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        print(f"[sim_capture] ffmpeg started: {w}x{h} at {x},{y}")
        return _ffmpeg_proc
    except FileNotFoundError:
        print("[sim_capture] ffmpeg not found. Install: sudo apt-get install -y ffmpeg")
        return None

# ─── Periodic window recheck ──────────────────────────────────────────────────

async def window_watcher():
    global _window_geom, _last_check, _ffmpeg_proc
    while True:
        await asyncio.sleep(5)
        geom = find_window_geom()
        if geom is None:
            if _window_geom is not None:
                print("[sim_capture] Window lost — stopping ffmpeg")
                if _ffmpeg_proc and _ffmpeg_proc.poll() is None:
                    _ffmpeg_proc.terminate()
                _ffmpeg_proc = None
                _window_geom = None
        else:
            moved = (_window_geom is None or
                     abs(geom["x"] - _window_geom["x"]) > 50 or
                     abs(geom["y"] - _window_geom["y"]) > 50)
            if moved:
                print(f"[sim_capture] Window moved/found — restarting ffmpeg")
                _window_geom = geom
                start_ffmpeg(geom)

# ─── MJPEG streaming ─────────────────────────────────────────────────────────

@app.get("/sim-stream")
async def sim_stream():
    geom = find_window_geom()
    if geom is None:
        async def placeholder():
            # Return a minimal "offline" JPEG as a single MJPEG frame
            # This is a 1x1 pixel green JPEG
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + _offline_jpeg()
                   + b"\r\n")
        return StreamingResponse(placeholder(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    global _window_geom, _ffmpeg_proc
    if _window_geom is None or _ffmpeg_proc is None or _ffmpeg_proc.poll() is not None:
        _window_geom = geom
        proc = start_ffmpeg(geom)
        if proc is None:
            return HTMLResponse("ffmpeg not available", status_code=503)
        _ffmpeg_proc = proc

    proc = _ffmpeg_proc

    async def stream():
        loop = asyncio.get_event_loop()
        try:
            while proc.poll() is None:
                chunk = await loop.run_in_executor(None, proc.stdout.read, 8192)
                if not chunk:
                    break
                yield chunk
        except Exception:
            pass

    return StreamingResponse(stream(),
                             media_type="multipart/x-mixed-replace; boundary=ffmpeg")

@app.get("/status")
async def status():
    geom = find_window_geom()
    return {
        "window_found": geom is not None,
        "geometry": geom,
        "ffmpeg_running": _ffmpeg_proc is not None and _ffmpeg_proc.poll() is None,
    }

def _offline_jpeg() -> bytes:
    """Tiny valid JPEG (10x10 dark green pixel)."""
    # A minimal 10x10 dark green JPEG blob
    return (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n'
        b'\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d'
        b'\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
        b'\xff\xc0\x00\x0b\x08\x00\n\x00\n\x01\x01\x11\x00\xff\xc4\x00\x1f\x00'
        b'\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02'
        b'\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00'
        b'\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1'
        b'\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789'
        b':;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9'
    )

# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(window_watcher())
    geom = find_window_geom()
    if geom:
        global _window_geom
        _window_geom = geom
        start_ffmpeg(geom)
        print(f"[sim_capture] MuJoCo window found at startup: {geom}")
    else:
        print("[sim_capture] No MuJoCo window found — will retry every 5s")
    print(f"[sim_capture] Serving at http://0.0.0.0:{args.port}/sim-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning", ws="wsproto")
