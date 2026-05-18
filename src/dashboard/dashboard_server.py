#!/usr/bin/env python3
"""
ESA Edge-Perception Mission Control Dashboard v2.
Runs on the Jetson HOST (not inside Docker).

pip3 install fastapi uvicorn wsproto

Usage:
    python3 dashboard_server.py [--host 0.0.0.0] [--port 8080]
                                [--meta-host 127.0.0.1] [--meta-port 8091]
                                [--lidar-host 127.0.0.1] [--lidar-port 8092]
                                [--mock]
"""

import argparse
import asyncio
import json
import math
import random
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# ─── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="ESA Mission Control Dashboard v2")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--video-host", default="127.0.0.1")
parser.add_argument("--meta-host", default="127.0.0.1")
parser.add_argument("--meta-port", type=int, default=8091)
parser.add_argument("--lidar-host", default="127.0.0.1")
parser.add_argument("--lidar-port", type=int, default=8092)
parser.add_argument("--mock", action="store_true", help="Synthetic data — no hardware needed")
args = parser.parse_args()

# ─── ConstrainedProxy (from network_test.py) ─────────────────────────────────

class ConstrainedProxy:
    """TCP proxy applying network constraints (delay/bandwidth/loss)."""

    def __init__(self, listen_port, dest_host, dest_port,
                 delay_ms=0, bandwidth_bps=0, loss_pct=0.0):
        self.listen_port = listen_port
        self.dest_host = dest_host
        self.dest_port = dest_port
        self.delay_ms = delay_ms
        self.bandwidth_bps = bandwidth_bps
        self.loss_pct = loss_pct
        self._stop = threading.Event()
        self._server = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._server:
            self._server.close()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", self.listen_port))
        self._server.listen(4)
        self._server.settimeout(1.0)
        while not self._stop.is_set():
            try:
                client_conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                upstream.connect((self.dest_host, self.dest_port))
            except (ConnectionRefusedError, OSError):
                client_conn.close()
                continue
            threading.Thread(target=self._forward, args=(upstream, client_conn, True), daemon=True).start()
            threading.Thread(target=self._forward, args=(client_conn, upstream, False), daemon=True).start()
        self._server.close()

    def _forward(self, src, dst, apply_constraints):
        try:
            src.settimeout(2.0)
            while not self._stop.is_set():
                try:
                    data = src.recv(65536)
                except socket.timeout:
                    continue
                if not data:
                    break
                if apply_constraints:
                    if self.loss_pct > 0 and random.random() * 100 < self.loss_pct:
                        continue
                    if self.delay_ms > 0:
                        time.sleep(self.delay_ms / 1000.0)
                    if self.bandwidth_bps > 0:
                        time.sleep(len(data) / self.bandwidth_bps)
                try:
                    dst.sendall(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        except Exception:
            pass
        finally:
            try: src.close()
            except OSError: pass
            try: dst.close()
            except OSError: pass

# ─── ConstrainedHTTPProxy (MJPEG-frame-aware) ────────────────────────────────

class ConstrainedHTTPProxy:
    """MJPEG-frame-aware HTTP proxy.

    Connects to an upstream multipart/x-mixed-replace MJPEG stream, parses
    --frame boundaries, extracts complete JPEG frames, and applies per-frame
    network constraints before re-emitting a valid MJPEG stream to clients.
    Byte-level throttling on a continuous HTTP stream doesn't reduce FPS;
    operating at the frame level does.
    """

    def __init__(self, listen_port, upstream_host, upstream_port,
                 upstream_path="/stream",
                 delay_ms=0, bandwidth_bps=0, loss_pct=0.0):
        self.listen_port = listen_port
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.upstream_path = upstream_path
        self.delay_ms = delay_ms
        self.bandwidth_bps = bandwidth_bps
        self.loss_pct = loss_pct
        self._stop = threading.Event()
        self._server = None
        self._thread = None
        self.frames_forwarded = 0
        self.frames_dropped = 0
        self._frame_times: deque = deque()  # timestamps of forwarded frames for fps calc

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._server:
            self._server.close()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def fps(self) -> float:
        ft = self._frame_times
        try:
            if len(ft) < 2:
                return 0.0
            span = ft[-1] - ft[0]
            return (len(ft) - 1) / span if span > 0 else 0.0
        except (IndexError, ZeroDivisionError):
            # Deque can shrink between len() check and index access (GIL is not
            # held across the pair of operations) — return 0 rather than crash.
            return 0.0

    def _run(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", self.listen_port))
        self._server.listen(4)
        self._server.settimeout(1.0)
        while not self._stop.is_set():
            try:
                client_conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_client, args=(client_conn,), daemon=True
            ).start()
        self._server.close()

    def _handle_client(self, client_conn):
        upstream_sock = None
        try:
            # Drain incoming client HTTP request
            client_conn.settimeout(5.0)
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = client_conn.recv(1024)
                if not chunk:
                    return
                buf += chunk

            # Open upstream MJPEG connection
            upstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            upstream_sock.settimeout(10.0)
            upstream_sock.connect((self.upstream_host, self.upstream_port))
            upstream_sock.sendall(
                f"GET {self.upstream_path} HTTP/1.0\r\n"
                f"Host: {self.upstream_host}\r\n\r\n".encode()
            )

            upstream_file = upstream_sock.makefile("rb")

            # Skip upstream HTTP response headers
            while True:
                line = upstream_file.readline()
                if line in (b"\r\n", b"\n", b""):
                    break

            # Send our response headers to the client
            client_conn.sendall(
                b"HTTP/1.0 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                b"Cache-Control: no-cache\r\n"
                b"\r\n"
            )

            # Parse and proxy frames one at a time
            while not self._stop.is_set():
                # Find the next --frame boundary line
                boundary = upstream_file.readline()
                if not boundary:
                    break
                if not boundary.strip().startswith(b"--"):
                    continue  # skip preamble / blank lines

                # Parse MIME headers for this part
                content_length = None
                while True:
                    header = upstream_file.readline()
                    stripped = header.strip()
                    if not stripped:
                        break  # blank line = end of part headers
                    if stripped.lower().startswith(b"content-length:"):
                        try:
                            content_length = int(stripped.split(b":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass

                if content_length is None:
                    continue  # can't read frame without knowing its size

                # Read exactly content_length bytes of JPEG data
                jpeg_bytes = upstream_file.read(content_length)
                if len(jpeg_bytes) < content_length:
                    break  # upstream closed mid-frame

                # Per-frame: propagation delay
                if self.delay_ms > 0:
                    time.sleep(self.delay_ms / 1000.0)

                # Per-frame: packet loss
                if self.loss_pct > 0 and random.random() * 100 < self.loss_pct:
                    self.frames_dropped += 1
                    continue

                # Per-frame: bandwidth pacing
                if self.bandwidth_bps > 0:
                    time.sleep(len(jpeg_bytes) / self.bandwidth_bps)

                # Re-emit a valid MJPEG part to the client
                out = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg_bytes)}\r\n".encode()
                    + b"\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )
                client_conn.sendall(out)
                self.frames_forwarded += 1
                # Record frame time for fps calculation (5-second sliding window)
                t = time.time()
                self._frame_times.append(t)
                cutoff = t - 5.0
                while self._frame_times and self._frame_times[0] < cutoff:
                    self._frame_times.popleft()

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception:
            pass
        finally:
            try:
                client_conn.close()
            except OSError:
                pass
            if upstream_sock:
                try:
                    upstream_sock.close()
                except OSError:
                    pass

# ─── Profile Maps ─────────────────────────────────────────────────────────────

SIM_PROFILE_MAP = {
    "baseline":      {"delay_ms": 0,    "bandwidth_bps": 0,              "loss_pct": 0.0},
    "good_5g":       {"delay_ms": 20,   "bandwidth_bps": 50_000_000 // 8, "loss_pct": 0.0},
    "congested_5g":  {"delay_ms": 50,   "bandwidth_bps": 10_000_000 // 8, "loss_pct": 0.0},
    "poor_5g":       {"delay_ms": 100,  "bandwidth_bps": 5_000_000 // 8,  "loss_pct": 1.0},
    "satellite_ntn": {"delay_ms": 600,  "bandwidth_bps": 1_000_000 // 8,  "loss_pct": 2.0},
    "degraded_ntn":  {"delay_ms": 1200, "bandwidth_bps": 500_000 // 8,    "loss_pct": 5.0},
}

SIM_PROFILE_LABELS = {
    "baseline":      "Baseline",
    "good_5g":       "Good 5G",
    "congested_5g":  "Congested 5G",
    "poor_5g":       "Poor 5G",
    "satellite_ntn": "Satellite NTN",
    "degraded_ntn":  "Degraded NTN",
}

KNOWN_PROFILES = {
    "Baseline":      {"fps": 28.4, "latency_ms": 1.0},
    "Good 5G":       {"fps": 28.5, "latency_ms": 21.0},
    "Congested 5G":  {"fps": 12.1, "latency_ms": 77.0},
    "Poor 5G":       {"fps": 6.4,  "latency_ms": 152.0},
    "Satellite NTN": {"fps": 1.2,  "latency_ms": 932.0},
    "Degraded NTN":  {"fps": 0.6,  "latency_ms": 1927.0},
}

# ─── Global State ─────────────────────────────────────────────────────────────

START_TIME = time.time()

meta_state = {
    "active": False, "fps": 0.0, "inference_ms": 0.0,
    "frame_number": 0, "jpeg_bytes": 0, "last_seen": 0.0, "detections": [],
    "frame_w": 640, "frame_h": 480,
}
lidar_state = {
    "active": False, "hz": 0.0, "scan_id": 0, "num_points": 0,
    "points": [], "forward_distance_mm": 0, "rotation_speed_deg_s": 0, "last_seen": 0.0,
}
system_state_hw = {"gpu_temp_c": 0.0, "gpu_status": "nominal"}

active_meta_port: int = args.meta_port
active_lidar_port: int = args.lidar_port
active_sim_profile: Optional[str] = None
sim_proxies: list = []
_video_reconnect_token: int = 0  # incremented on profile change to force /video reconnect
meta_reader_task: Optional[asyncio.Task] = None
lidar_reader_task: Optional[asyncio.Task] = None
go2_api_proc = None

meta_ts_window: deque = deque()
lidar_ts_window: deque = deque()
frame_window: deque = deque()   # (ts, frame_number, jpeg_bytes)
lidar_bw_window: deque = deque()  # (ts, num_points)
session_totals: dict = {}
logging_enabled: bool = False
ai_stream_enabled: bool = True
_baseline_fps: float = 28.0
_video_output_enabled: bool = True
ws_clients: set = set()

network_state = {
    "video_bw_kbps": 0.0, "lidar_bw_kbps": 0.0, "meta_bw_kbps": 0.0,
    "total_bw_kbps": 0.0, "e2e_latency_ms": 0.0, "frame_loss_pct": 0.0,
    "quality_score": 0, "inferred_profile": "Unknown", "profile_confidence": 0.0,
}

app = FastAPI()

# ─── GPU Temperature ──────────────────────────────────────────────────────────

def read_gpu_temp() -> float:
    try:
        base = Path("/sys/devices/virtual/thermal/")
        for zone in sorted(base.glob("thermal_zone*")):
            if (zone / "type").read_text().strip() == "gpu-thermal":
                return int((zone / "temp").read_text().strip()) / 1000.0
    except Exception:
        pass
    return 0.0

def gpu_status_label(temp: float) -> str:
    if temp > 95: return "critical"
    if temp > 85: return "hot"
    if temp > 70: return "warm"
    return "nominal"

async def gpu_temp_poller():
    while True:
        t = read_gpu_temp()
        system_state_hw["gpu_temp_c"] = t
        system_state_hw["gpu_status"] = gpu_status_label(t)
        await asyncio.sleep(2)

# ─── FPS/Hz Sliding Window ────────────────────────────────────────────────────

def record_message(window: deque) -> float:
    now = time.time()
    window.append(now)
    cutoff = now - 2.0
    while window and window[0] < cutoff:
        window.popleft()
    return len(window) / 2.0

# ─── TCP Helpers ─────────────────────────────────────────────────────────────

async def recv_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = await reader.read(n - len(buf))
        if not chunk:
            raise ConnectionError("stream closed")
        buf += chunk
    return buf

async def read_framed_message(reader: asyncio.StreamReader) -> dict:
    header = await recv_exactly(reader, 4)
    length = struct.unpack(">I", header)[0]
    payload = await recv_exactly(reader, length)
    return json.loads(payload.decode("utf-8"))

# ─── Profile Inference ────────────────────────────────────────────────────────

def infer_profile(fps: float, latency_ms: float) -> tuple:
    max_fps, max_lat = 30.0, 2000.0
    nf = fps / max_fps
    nl = min(latency_ms, max_lat) / max_lat
    best_name, best_dist = "Baseline", float("inf")
    for name, p in KNOWN_PROFILES.items():
        pf = p["fps"] / max_fps
        pl = min(p["latency_ms"], max_lat) / max_lat
        dist = math.sqrt((nf - pf) ** 2 + (nl - pl) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    confidence = max(0.0, 1.0 - best_dist / math.sqrt(2))
    return best_name, round(confidence, 2)

# ─── Network Metrics ──────────────────────────────────────────────────────────

def compute_network_metrics():
    now = time.time()
    cutoff = now - 10.0
    while frame_window and frame_window[0][0] < cutoff:
        frame_window.popleft()
    while lidar_bw_window and lidar_bw_window[0][0] < cutoff:
        lidar_bw_window.popleft()

    # Zero out contributions from dead streams so the meter drops immediately.
    m_last = now - meta_state["last_seen"] if meta_state["last_seen"] else 999.0
    m_alive = m_last < 5.0 and meta_state["active"]
    l_last = now - lidar_state["last_seen"] if lidar_state["last_seen"] else 999.0
    l_alive = l_last < 5.0 and lidar_state["active"]

    fps = meta_state["fps"] if m_alive else 0.0
    jpeg_bytes = meta_state["jpeg_bytes"] if m_alive else 0
    hz = lidar_state["hz"] if l_alive else 0.0
    num_points = lidar_state["num_points"] if l_alive else 0

    video_bw_kbps = (jpeg_bytes * fps) / 1000.0 if fps > 0 else 0.0
    lidar_bw_kbps = (num_points * 12 * hz) / 1000.0
    meta_bw_kbps = (200.0 * fps) / 1000.0 if ai_stream_enabled else 0.0
    total_bw_kbps = video_bw_kbps + lidar_bw_kbps + meta_bw_kbps

    e2e_latency_ms = meta_state["inference_ms"]

    if len(frame_window) >= 2:
        span = frame_window[-1][0] - frame_window[0][0]
        if span > 0.5 and fps > 0:
            expected = fps * span
            received = len(frame_window)
            frame_loss_pct = max(0.0, min(100.0, (1.0 - received / max(expected, 1)) * 100.0))
        else:
            frame_loss_pct = 0.0
    else:
        frame_loss_pct = 0.0

    global _baseline_fps
    if active_sim_profile:
        # When sim is active, derive metrics from the video proxy (actual
        # delivered frame rate) and the configured profile delay, not the
        # metadata stream which barely degrades.
        proxy_fps = next(
            (p.fps for p in sim_proxies if isinstance(p, ConstrainedHTTPProxy)), 0.0)
        profile_delay = SIM_PROFILE_MAP.get(active_sim_profile, {}).get("delay_ms", 0)
        e2e_latency_ms = profile_delay + meta_state["inference_ms"]
        quality_score = round(max(0.0, min(100.0, (proxy_fps / _baseline_fps) * 100.0)))
        inferred = SIM_PROFILE_LABELS.get(active_sim_profile, "Unknown")
        confidence = 1.0
    else:
        if fps > 5.0:
            _baseline_fps = 0.9 * _baseline_fps + 0.1 * fps
        fps_score = min(fps / 30.0, 1.0) * 100
        latency_score = max(0.0, 100.0 - e2e_latency_ms / 10.0)
        delivery_score = (1.0 - frame_loss_pct / 100.0) * 100.0
        quality_score = round(0.4 * fps_score + 0.3 * latency_score + 0.3 * delivery_score)
        inferred, confidence = infer_profile(fps, e2e_latency_ms)

    network_state.update({
        "video_bw_kbps": round(video_bw_kbps, 1),
        "lidar_bw_kbps": round(lidar_bw_kbps, 1),
        "meta_bw_kbps": round(meta_bw_kbps, 1),
        "total_bw_kbps": round(total_bw_kbps, 1),
        "e2e_latency_ms": round(e2e_latency_ms, 1),
        "frame_loss_pct": round(frame_loss_pct, 2),
        "quality_score": quality_score,
        "inferred_profile": inferred,
        "profile_confidence": confidence,
    })

# ─── Meta TCP Reader ──────────────────────────────────────────────────────────

async def meta_reader():
    global active_meta_port
    backoff = 0.5
    while True:
        port = active_meta_port
        try:
            reader, _ = await asyncio.open_connection(args.meta_host, port)
            backoff = 0.5
            meta_state["active"] = True
            print(f"[meta] Connected {args.meta_host}:{port}")
            while True:
                msg = await read_framed_message(reader)
                hz = record_message(meta_ts_window)
                now = time.time()
                frame_window.append((now, msg.get("frame_number", 0), msg.get("jpeg_bytes", 0)))
                meta_state.update({
                    "active": True, "fps": hz,
                    "inference_ms": msg.get("inference_time_ms", 0.0),
                    "frame_number": msg.get("frame_number", 0),
                    "jpeg_bytes": msg.get("jpeg_bytes", 0),
                    "last_seen": time.time(),
                    "detections": msg.get("detections", []),
                    "send_timestamp":msg.get("send_timestamp", 0.0),
                    "frame_w": msg.get("frame_w", meta_state.get("frame_w", 640)),
                    "frame_h": msg.get("frame_h", meta_state.get("frame_h", 480)),
                })
                for det in msg.get("detections", []):
                    cls = det.get("class", "unknown")
                    session_totals[cls] = session_totals.get(cls, 0) + 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            meta_state["active"] = False
            print(f"[meta] Disconnected ({e}), retry in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

# ─── LiDAR TCP Reader ─────────────────────────────────────────────────────────

async def lidar_reader():
    global active_lidar_port
    backoff = 0.5
    while True:
        port = active_lidar_port
        try:
            reader, _ = await asyncio.open_connection(args.lidar_host, port)
            backoff = 0.5
            lidar_state["active"] = True
            print(f"[lidar] Connected {args.lidar_host}:{port}")
            while True:
                msg = await read_framed_message(reader)
                hz = record_message(lidar_ts_window)
                points = msg.get("points", [])
                fwd = [p[1] for p in points if p[1] > 0 and (p[0] <= 15 or p[0] >= 345)]
                lidar_bw_window.append((time.time(), msg.get("num_points", 0)))
                lidar_state.update({
                    "active": True, "hz": hz,
                    "scan_id": msg.get("scan_id", 0),
                    "num_points": msg.get("num_points", 0),
                    "points": points,
                    "forward_distance_mm": int(sum(fwd) / len(fwd)) if fwd else 0,
                    "rotation_speed_deg_s": msg.get("rotation_speed_deg_s", 0),
                    "last_seen": time.time(),
                })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            lidar_state["active"] = False
            print(f"[lidar] Disconnected ({e}), retry in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

# ─── Restart Stream Readers ───────────────────────────────────────────────────

async def restart_stream_readers():
    global meta_reader_task, lidar_reader_task
    for task in (meta_reader_task, lidar_reader_task):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    meta_reader_task = asyncio.create_task(meta_reader())
    lidar_reader_task = asyncio.create_task(lidar_reader())

# ─── Mock Generator ───────────────────────────────────────────────────────────

async def mock_generator():
    scan_id = 0
    frame_number = 0
    gpu_cycle = 0.0
    net_cycle = 0.0
    classes = ["person", "car", "bicycle", "dog", "truck"]
    while True:
        now = time.time()
        scan_id += 1
        frame_number += random.randint(1, 3)
        gpu_cycle += 0.04
        net_cycle += 0.017  # ~60s full cycle

        gpu_temp = 70 + 20 * math.sin(gpu_cycle)
        system_state_hw["gpu_temp_c"] = round(gpu_temp, 1)
        system_state_hw["gpu_status"] = gpu_status_label(gpu_temp)

        # Network oscillation: 0-1 cycle maps to baseline→degraded NTN→baseline
        t = (math.sin(net_cycle) + 1) / 2  # 0..1
        mock_fps = 0.6 + (28.4 - 0.6) * (1 - t)
        mock_lat = 1.0 + (1927.0 - 1.0) * t
        mock_loss = t * 8.0
        mock_bw = 20.0 + (1200.0 - 20.0) * (1 - t)

        fps_score = min(mock_fps / 30.0, 1.0) * 100
        lat_score = max(0.0, 100.0 - mock_lat / 10.0)
        del_score = (1.0 - mock_loss / 100.0) * 100.0
        mock_quality = round(0.4 * fps_score + 0.3 * lat_score + 0.3 * del_score)
        mock_inferred, mock_conf = infer_profile(mock_fps, mock_lat)

        network_state.update({
            "video_bw_kbps": round(mock_bw * 0.82, 1),
            "lidar_bw_kbps": round(mock_bw * 0.17, 1),
            "meta_bw_kbps": round(mock_bw * 0.01, 1),
            "total_bw_kbps": round(mock_bw, 1),
            "e2e_latency_ms": round(mock_lat, 1),
            "frame_loss_pct": round(mock_loss, 2),
            "quality_score": mock_quality,
            "inferred_profile": mock_inferred,
            "profile_confidence": mock_conf,
        })

        fps = 14 + 14.5 * abs(math.sin(now / 20))
        n_dets = random.randint(0, 4)
        detections = []
        for _ in range(n_dets):
            cls = random.choice(classes)
            conf = round(random.uniform(0.5, 0.99), 2)
            cx = random.randint(80, 560)
            cy = random.randint(60, 420)
            bw = random.randint(40, 180)
            bh = random.randint(50, 200)
            detections.append({
                "class": cls, "confidence": conf,
                "bbox": [max(0, cx - bw // 2), max(0, cy - bh // 2),
                         min(640, cx + bw // 2), min(480, cy + bh // 2)],
            })
            session_totals[cls] = session_totals.get(cls, 0) + 1

        meta_state.update({
            "active": True, "fps": round(fps, 1),
            "inference_ms": round(random.uniform(8, 25), 1),
            "frame_number": frame_number,
            "jpeg_bytes": random.randint(30000, 60000),
            "last_seen": now, "detections": detections,
            "send_timestamp": now,
            "frame_w": 640, "frame_h": 480,
        })

        points = []
        for angle_i in range(0, 360, 1):
            if 45 <= angle_i <= 55:
                dist = int(800 + random.gauss(0, 30))
            elif 200 <= angle_i <= 210:
                dist = int(1500 + random.gauss(0, 50))
            else:
                dist = int(3000 + random.gauss(0, 200))
            dist = max(100, min(6000, dist))
            points.append([float(angle_i), dist, random.randint(150, 255)])

        fwd = [p[1] for p in points if p[0] <= 15 or p[0] >= 345]
        lidar_state.update({
            "active": True, "hz": round(random.uniform(9.0, 10.5), 1),
            "scan_id": scan_id, "num_points": len(points), "points": points,
            "forward_distance_mm": int(sum(fwd) / len(fwd)) if fwd else 0,
            "rotation_speed_deg_s": 4500, "last_seen": now,
        })
        record_message(meta_ts_window)
        record_message(lidar_ts_window)
        await asyncio.sleep(0.1)

# ─── Telemetry Builder ────────────────────────────────────────────────────────

def build_telemetry() -> dict:
    now = time.time()
    m_last = now - meta_state["last_seen"] if meta_state["last_seen"] else 999.0
    l_last = now - lidar_state["last_seen"] if lidar_state["last_seen"] else 999.0

    # Track each stream independently:
    #   cam_alive  — derived from metadata: camera is alive when metadata flows
    #                (TCP probe on 8090 was removed — the MJPEG HTTPServer's tiny
    #                 backlog fills up under probe load, starving real connections)
    #   m_alive    — metadata stream on port 8091 is delivering frames
    #   l_alive    — LiDAR stream on port 8092 is delivering scans
    m_alive = m_last < 5.0 and meta_state["active"]
    cam_alive = True if args.mock else m_alive
    l_alive = l_last < 5.0 and lidar_state["active"]

    ai_alive = ai_stream_enabled
    video_alive = cam_alive and _video_output_enabled
    n_alive = sum([video_alive, ai_alive, l_alive])
    if n_alive == 3:
        sys_state = "online"
    elif n_alive > 0:
        sys_state = "degraded"
    else:
        sys_state = "offline"

    if not args.mock:
        compute_network_metrics()

    if ai_stream_enabled:
        det_items = [
            {k: d[k] for k in ("class", "confidence", "bbox") if k in d}
            for d in meta_state["detections"]
        ]
        det_classes: dict = {}
        for d in det_items:
            det_classes[d["class"]] = det_classes.get(d["class"], 0) + 1
    else:
        det_items = []
        det_classes = {}

    active_label = SIM_PROFILE_LABELS.get(active_sim_profile) if active_sim_profile else None

    # meta_fps: raw metadata delivery rate (always ~30; survives degraded links
    # because metadata messages are tiny). video_fps: actual video frame rate
    # the browser sees (throttled by proxy when sim is active).
    meta_fps = (meta_state["fps"] if m_alive else 0.0) if ai_stream_enabled else 0.0
    if active_sim_profile:
        video_fps = next(
            (p.fps for p in sim_proxies if isinstance(p, ConstrainedHTTPProxy)), 0.0
        )
    else:
        video_fps = meta_fps

    return {
        "type": "telemetry",
        "timestamp": now,
        "system_state": sys_state,
        "active_network_profile": active_label,
        "camera": {
            "active": cam_alive,
            "fps": video_fps,
            "meta_fps": meta_fps,
            "inference_ms": meta_state["inference_ms"],
            "frame_number": meta_state["frame_number"],
            "jpeg_bytes": meta_state["jpeg_bytes"],
            "last_seen_s": round(m_last, 2),
            "send_timestamp": meta_state.get("send_timestamp", 0.0),
            "frame_w": meta_state.get("frame_w", 640),
            "frame_h": meta_state.get("frame_h", 480),
        },
        "detections": {"count": len(det_items), "classes": det_classes, "items": det_items},
        "lidar": {
            "active": lidar_state["active"],
            "hz": lidar_state["hz"] if l_alive else 0.0,
            "scan_id": lidar_state["scan_id"],
            "num_points": lidar_state["num_points"],
            "points": lidar_state["points"],
            "forward_distance_mm": lidar_state["forward_distance_mm"],
            "rotation_speed_deg_s": lidar_state["rotation_speed_deg_s"],
            "last_seen_s": round(l_last, 2),
        },
        "system": {
            "gpu_temp_c": system_state_hw["gpu_temp_c"],
            "gpu_status": system_state_hw["gpu_status"],
            "uptime_s": round(now - START_TIME, 1),
        },
        "streams": {
            "video_alive": video_alive,
            "ai_alive": ai_alive,       # AI SA stream (toggle on = alive)
            "lidar_alive": l_alive,     # LiDAR stream port 8092
        },
        "network": dict(network_state),
        "session_totals": session_totals,
        "ai_stream_enabled": ai_stream_enabled,
    }

# ─── WebSocket ────────────────────────────────────────────────────────────────

async def telemetry_broadcaster():
    while True:
        if ws_clients:
            msg = json.dumps(build_telemetry())
            dead = set()
            for ws in list(ws_clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            ws_clients.difference_update(dead)
        await asyncio.sleep(0.1)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)

# ─── HTTP Routes ──────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    return HTMLResponse(DASHBOARD_HTML)

@app.get("/health")
async def health():
    return {"status": "ok", "uptime_s": round(time.time() - START_TIME, 1)}

@app.get("/video")
async def proxy_video():
    """MJPEG proxy with persistent reconnect. Routes through ConstrainedHTTPProxy
    on 9090 when sim is active, otherwise connects directly to stream_server on
    8090. Outer loop reconnects on any mid-stream drop so the browser never sees
    a permanent EOF while the upstream is healthy."""

    async def stream_mjpeg():
        while True:  # persistent reconnect — exits only when browser disconnects
            video_port = 9090 if active_sim_profile else 8090
            token = _video_reconnect_token  # snapshot; break if profile changes
            writer = None
            # Inner retry loop: allows proxy thread time to bind after profile switch.
            for attempt in range(10):
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection("127.0.0.1", video_port), timeout=0.5)
                    break
                except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                    if attempt == 9:
                        break
                    if attempt > 0:
                        await asyncio.sleep(0.2)
            if writer is None:
                # Upstream not ready — pause before outer retry.
                await asyncio.sleep(1.0)
                continue
            try:
                writer.write(b"GET /stream HTTP/1.0\r\nHost: localhost\r\n\r\n")
                await writer.drain()
                # Read upstream HTTP response headers.
                header_buf = b""
                while b"\r\n\r\n" not in header_buf:
                    chunk = await asyncio.wait_for(reader.read(512), timeout=5.0)
                    if not chunk:
                        break
                    header_buf += chunk
                if b"\r\n\r\n" not in header_buf:
                    # Incomplete headers — upstream closed early.
                    await asyncio.sleep(1.0)
                    continue
                sep = header_buf.find(b"\r\n\r\n") + 4
                if sep < len(header_buf):
                    yield header_buf[sep:]
                # Stream body until connection drops or profile changes.
                while True:
                    if _video_reconnect_token != token:
                        break  # sim profile changed — reconnect to new port
                    chunk = await asyncio.wait_for(reader.read(8192), timeout=10.0)
                    if not chunk:
                        break
                    yield chunk
            except (asyncio.TimeoutError, ConnectionResetError, OSError):
                pass
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
            # Brief pause before reconnect to avoid tight loop on persistent failure.
            await asyncio.sleep(0.5)

    return StreamingResponse(stream_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

async def _port_open(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=1.0)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False

async def _http_get(host: str, port: int, path: str, timeout: float = 2.0) -> bool:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout)
        writer.write(f"GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
        await writer.drain()
        resp = await asyncio.wait_for(reader.read(256), timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return b"200" in resp[:32]
    except Exception:
        return False

@app.get("/control/status")
async def control_status():
    cam = await _port_open("127.0.0.1", 8090)
    lidar = await _port_open("127.0.0.1", 8092)
    return {"camera": "running" if cam else "stopped",
            "lidar": "running" if lidar else "stopped",
            "logging": logging_enabled}

@app.post("/control/camera/start")
async def camera_start():
    global _video_output_enabled
    if args.mock:
        _video_output_enabled = True
        return {"ok": True}
    meta_up = await _port_open("127.0.0.1", 8091)
    if meta_up:
        await _http_get("127.0.0.1", 8095, "/video/enable")
    else:
        await asyncio.create_subprocess_exec(
            "docker", "exec", "-d", "yolo-saad", "python3", "/workspace/stream_server.py")
    _video_output_enabled = True
    return {"ok": True}

@app.post("/control/camera/stop")
async def camera_stop():
    global _video_output_enabled
    if args.mock:
        _video_output_enabled = False
        return {"ok": True}
    await _http_get("127.0.0.1", 8095, "/video/disable")
    _video_output_enabled = False
    return {"ok": True}

@app.post("/control/lidar/start")
async def lidar_start():
    if args.mock:
        return {"ok": True}
    await asyncio.create_subprocess_exec(
        "python3", str(Path.home() / "Desktop" / "lidar_stream.py"))
    return {"ok": True}

@app.post("/control/lidar/stop")
async def lidar_stop():
    if args.mock:
        return {"ok": True}
    p = await asyncio.create_subprocess_exec("pkill", "-f", "lidar_stream.py")
    await p.wait()
    return {"ok": True}

@app.post("/control/logging/start")
async def logging_start():
    global logging_enabled
    logging_enabled = True
    return {"ok": True}

@app.post("/control/logging/stop")
async def logging_stop():
    global logging_enabled
    logging_enabled = False
    return {"ok": True}

@app.post("/control/aistream/start")
async def aistream_start():
    global ai_stream_enabled
    ai_stream_enabled = True
    return {"ok": True}

@app.post("/control/aistream/stop")
async def aistream_stop():
    global ai_stream_enabled
    ai_stream_enabled = False
    return {"ok": True}

# ─── Network Simulation ───────────────────────────────────────────────────────

@app.post("/control/network/start/{profile}")
async def network_sim_start(profile: str):
    global active_sim_profile, sim_proxies, active_meta_port, active_lidar_port, _video_reconnect_token
    if profile not in SIM_PROFILE_MAP:
        return JSONResponse({"ok": False, "error": f"Unknown profile: {profile}"}, status_code=400)

    for p in sim_proxies:
        p.stop()
    sim_proxies.clear()

    params = SIM_PROFILE_MAP[profile]
    vp = ConstrainedHTTPProxy(9090, "127.0.0.1", 8090, "/stream", **params)
    mp = ConstrainedProxy(9091, "127.0.0.1", args.meta_port, **params)
    lp = ConstrainedProxy(9092, "127.0.0.1", args.lidar_port, **params)
    vp.start()
    mp.start()
    lp.start()
    sim_proxies.extend([vp, mp, lp])

    active_sim_profile = profile
    active_meta_port = 9091
    active_lidar_port = 9092
    _video_reconnect_token += 1  # signal /video to reconnect via proxy

    if not args.mock:
        await restart_stream_readers()

    return {"ok": True, "profile": SIM_PROFILE_LABELS[profile]}

@app.post("/control/network/stop")
async def network_sim_stop():
    global active_sim_profile, sim_proxies, active_meta_port, active_lidar_port, _video_reconnect_token
    for p in sim_proxies:
        p.stop()
    sim_proxies.clear()
    active_sim_profile = None
    active_meta_port = args.meta_port
    active_lidar_port = args.lidar_port
    _video_reconnect_token += 1  # signal /video to reconnect directly to 8090
    if not args.mock:
        await restart_stream_readers()
    return {"ok": True}

@app.get("/control/network/status")
async def network_sim_status():
    return {
        "active": active_sim_profile is not None,
        "profile": SIM_PROFILE_LABELS.get(active_sim_profile) if active_sim_profile else None,
        "profile_key": active_sim_profile,
    }

# ─── Go2 Proxy ───────────────────────────────────────────────────────────────

async def _go2_request(method: str, path: str) -> dict:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", 8094), timeout=3.0)
        req = f"{method} {path} HTTP/1.0\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"
        writer.write(req.encode())
        await writer.drain()
        response = b""
        while True:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            if not chunk:
                break
            response += chunk
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        if b"\r\n\r\n" in response:
            return json.loads(response.split(b"\r\n\r\n", 1)[1])
        return {"success": False, "message": "Empty response"}
    except Exception as e:
        return {"success": False, "message": f"go2_api offline: {e}"}

@app.get("/go2/status")
async def go2_status():
    if args.mock:
        return {"target": "sim", "sdk_connected": True, "last_command": "none"}
    return await _go2_request("GET", "/status")

@app.post("/go2/{command}")
async def go2_command(command: str):
    if args.mock:
        return {"success": True, "command": command, "message": f"Mock: {command} simulated"}
    return await _go2_request("POST", f"/command/{command}")

@app.post("/go2-target/{target}")
async def go2_set_target(target: str):
    global go2_api_proc
    if target not in ("sim", "robot"):
        return JSONResponse({"ok": False, "error": "target must be sim or robot"}, status_code=400)
    if go2_api_proc and go2_api_proc.returncode is None:
        go2_api_proc.terminate()
        try:
            await asyncio.wait_for(go2_api_proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            go2_api_proc.kill()
    go2_api_proc = await asyncio.create_subprocess_exec(
        "python3", str(Path.home() / "Desktop" / "go2_api.py"),
        "--target", target,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return {"ok": True, "target": target}

# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global meta_reader_task, lidar_reader_task
    if args.mock:
        print("[dashboard] Mock mode — synthetic telemetry")
        asyncio.create_task(mock_generator())
    else:
        meta_reader_task = asyncio.create_task(meta_reader())
        lidar_reader_task = asyncio.create_task(lidar_reader())
        asyncio.create_task(gpu_temp_poller())
    asyncio.create_task(telemetry_broadcaster())
    print(f"[dashboard] Serving at http://{args.host}:{args.port}")

# ─── Dashboard HTML ───────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FENRIR-1 MISSION CONTROL</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
:root {
  --bg:      #0d0f0e;
  --bgp:     #111814;
  --bgpa:    #0a0f0c;
  --bdr:     #1e3028;
  --grn:     #00ff88;
  --grnd:    #00aa55;
  --grnk:    #005522;
  --amb:     #ffaa00;
  --ambd:    #aa6600;
  --red:     #ff3333;
  --txt:     #c8ffd4;
  --txtd:    #4a7a5a;
  --txtm:    #2a4a35;
}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{width:100vw;height:100vh;overflow:hidden;}
body{
  background:var(--bg);color:var(--txt);
  font-family:'JetBrains Mono','Courier New',monospace;
  display:flex;flex-direction:column;
}
body::after{
  content:'';position:fixed;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(to bottom,rgba(0,20,10,.12) 0,rgba(0,20,10,.12) 1px,transparent 1px,transparent 3px);
}
/* Status bar */
#sbar{
  height:28px;flex-shrink:0;background:var(--bgpa);border-bottom:1px solid var(--bdr);
  display:flex;align-items:center;padding:0 12px;gap:0;
  font-size:.60rem;color:var(--txtd);letter-spacing:.08em;white-space:nowrap;overflow:hidden;
}
#sbar .sep{margin:0 8px;color:var(--txtm);}
/* Offline banner */
#offlineBanner{
  flex-shrink:0;background:#2a1500;border-bottom:2px solid var(--amb);
  display:none;align-items:center;justify-content:center;
  font-size:.72rem;color:var(--amb);letter-spacing:.18em;text-transform:uppercase;
  padding:4px 0;text-align:center;
}
@keyframes blink{50%{opacity:0;}}
.blink{animation:blink 1s step-end infinite;}
.blink2{animation:blink 2s step-end infinite;}
/* Degraded bar */
#degradedBar{
  flex-shrink:0;background:#1a1200;border-bottom:1px solid var(--ambd);
  display:none;align-items:center;justify-content:center;
  font-size:.58rem;color:var(--amb);letter-spacing:.12em;padding:2px 0;
}
/* Main */
#main{flex:1;min-height:0;display:flex;flex-direction:column;gap:3px;padding:3px;}
#topRow{flex:55;min-height:0;display:flex;gap:3px;}
#botRow{flex:45;min-height:0;display:flex;gap:3px;}
.panel{background:var(--bgp);border:1px solid var(--bdr);display:flex;flex-direction:column;overflow:hidden;position:relative;}
.ph{
  height:22px;flex-shrink:0;background:var(--bgpa);border-bottom:1px solid var(--bdr);
  display:flex;align-items:center;padding:0 8px;
  font-size:.58rem;color:var(--txtd);letter-spacing:.13em;text-transform:uppercase;
}
.pb{flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0;}
/* Camera */
#pCam{flex:3;}
#vcont{flex:1;position:relative;background:#000;overflow:hidden;}
#vfeed{width:100%;height:100%;object-fit:contain;display:block;}
#nosig{
  display:flex;position:absolute;top:0;left:0;right:0;bottom:0;
  background:#080a09;align-items:center;justify-content:center;
  color:var(--amb);font-size:1.1rem;letter-spacing:.2em;
}
#livebadge{position:absolute;top:6px;right:8px;font-size:.60rem;color:var(--grn);display:flex;align-items:center;gap:3px;}
#oBL{position:absolute;bottom:6px;left:8px;font-size:.62rem;color:var(--grn);background:rgba(0,0,0,.55);padding:1px 5px;}
#oBR{position:absolute;bottom:6px;right:8px;font-size:.62rem;color:var(--grn);background:rgba(0,0,0,.55);padding:1px 5px;}
/* LiDAR */
#pLidar{flex:1.5;}
/* Sim View */
#pSim{flex:1.2;display:none;}
#simFeedWrap{flex:1;position:relative;background:#000;overflow:hidden;}
#simFeed{width:100%;height:100%;object-fit:contain;display:block;}
#simOffline{
  display:flex;position:absolute;top:0;left:0;right:0;bottom:0;
  background:#040804;align-items:center;justify-content:center;
  color:var(--txtd);font-size:.70rem;letter-spacing:.15em;flex-direction:column;gap:4px;
}
/* Controls */
#pCtrl{flex:1.5;}
#pCtrl .pb{padding:6px 7px;display:flex;flex-direction:column;gap:4px;font-size:.62rem;overflow-y:auto;}
.csec{border-top:1px solid var(--bdr);padding-top:4px;margin-top:2px;}
.csec:first-child{border-top:none;padding-top:0;}
.clbl{color:var(--txtd);font-size:.55rem;letter-spacing:.1em;margin-bottom:2px;}
.crow{display:flex;align-items:center;justify-content:space-between;gap:4px;}
.sdot{font-size:.80rem;}
.sdot.g{color:var(--grn);}.sdot.a{color:var(--amb);}.sdot.r{color:var(--red);}.sdot.d{color:var(--txtm);}
.cbtn{
  background:transparent;border:1px solid var(--grnd);color:var(--txt);
  font-family:inherit;font-size:.57rem;padding:2px 6px;cursor:pointer;letter-spacing:.04em;
}
.cbtn:hover{background:var(--grnk);}
.cbtn.stp{border-color:var(--ambd);}
.cbtn.stp:hover{background:#2a1200;}
.cbtn.dim{border-color:var(--ambd);color:var(--ambd);opacity:.5;cursor:not-allowed;}
.cbtn.act{border-color:var(--grn);background:var(--grnk);}
/* profile grid */
.pgrid{display:grid;grid-template-columns:1fr 1fr;gap:2px;margin-top:3px;}
.pbtn{
  background:transparent;border:1px solid var(--bdr);color:var(--txtd);
  font-family:inherit;font-size:.52rem;padding:2px 4px;cursor:pointer;text-align:center;
  letter-spacing:.02em;
}
.pbtn:hover{border-color:var(--grnd);color:var(--txt);}
.pbtn.pact{border-color:var(--grn);background:var(--grnk);color:var(--grn);}
.pbtn.pact-ntn{border-color:var(--red);background:#1a0000;color:var(--red);}
/* Robot control */
.go2btn{
  background:transparent;border:1px solid var(--grnd);color:var(--txt);
  font-family:inherit;font-size:.57rem;padding:3px 6px;cursor:pointer;flex:1;text-align:center;
}
.go2btn:hover{background:var(--grnk);}
.go2btn.flash{background:var(--grnk);border-color:var(--grn);}
.go2btn.unimpl{border-color:var(--ambd);color:var(--ambd);opacity:.5;cursor:not-allowed;}
.tgt{
  background:transparent;border:1px solid var(--bdr);color:var(--txtd);
  font-family:inherit;font-size:.55rem;padding:1px 8px;cursor:pointer;
}
.tgt.sel{border-color:var(--grn);background:var(--grnk);color:var(--grn);}
#eStop{
  width:100%;margin-top:auto;border:1px solid var(--red);color:var(--amb);
  font-family:inherit;font-size:.60rem;background:transparent;padding:5px;cursor:pointer;
}
#eStop:hover{background:#1a0000;}
/* Detections */
#pDet{flex:1.2;}
#pDet .pb{padding:6px 7px;font-size:.62rem;overflow-y:auto;display:flex;flex-direction:column;gap:2px;}
.dstat .lbl{color:var(--txtd);}.dstat .val{color:var(--grn);font-size:.78rem;}
.ditem{display:flex;align-items:center;gap:5px;margin-top:2px;}
.dcls{color:var(--txt);width:60px;overflow:hidden;}
.dbar{font-size:.72rem;}
.dconf{color:var(--txtd);width:30px;font-size:.60rem;}
.sdiv{border-top:1px solid var(--bdr);margin:5px 0 3px;color:var(--txtd);font-size:.55rem;letter-spacing:.08em;padding-top:3px;}
.sitem{display:flex;justify-content:space-between;color:var(--txtd);font-size:.60rem;margin-bottom:1px;}
/* AI Situational Awareness */
#pAiSa{flex:2.5;}
#aiWrap{flex:1;position:relative;background:#0d0f0e;overflow:hidden;min-height:0;}
#aiCanvas{width:100%;height:100%;display:block;}
#aiOffline{
  display:none;position:absolute;top:0;left:0;right:0;bottom:0;
  background:#0d0f0e;align-items:center;justify-content:center;
  color:var(--amb);font-size:.80rem;letter-spacing:.18em;
}
#aiTicker{
  flex-shrink:0;height:90px;overflow:hidden;padding:3px 6px;
  border-top:1px solid var(--bdr);font-size:.55rem;
}
.aiTick{opacity:0;animation:tickIn .3s forwards;margin-bottom:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
@keyframes tickIn{to{opacity:1;}}
.aiTick .ts{color:var(--txtd);}
.aiTick .cnt{color:var(--grn);}
#aiRate{
  flex-shrink:0;display:flex;gap:6px;padding:4px 6px;border-top:1px solid var(--bdr);
  font-size:.55rem;color:var(--txtd);letter-spacing:.06em;
}
.aiBar{flex:1;display:flex;flex-direction:column;gap:1px;}
.aiBarLbl{display:flex;justify-content:space-between;}
.aiBarTrack{height:6px;background:#111;border:1px solid var(--bdr);overflow:hidden;}
.aiBarFill{height:100%;transition:width .3s;}
/* Network panel */
#pNet{flex:4;}
#pNet .pb{flex-direction:row;padding:0;align-items:stretch;}
.nsp{
  border-right:1px solid var(--bdr);padding:6px 7px;
  display:flex;flex-direction:column;gap:3px;overflow:hidden;min-width:0;
}
.nsp:last-child{border-right:none;}
.nlbl{font-size:.52rem;color:var(--txtd);letter-spacing:.1em;text-transform:uppercase;flex-shrink:0;}
.nval{font-size:1.1rem;font-variant-numeric:tabular-nums;line-height:1.1;}
.nval.g{color:var(--grn);}.nval.a{color:var(--amb);}.nval.r{color:var(--red);}
/* BW bar */
.bwtrack{height:10px;display:flex;border:1px solid var(--bdr);overflow:hidden;border-radius:1px;flex-shrink:0;}
.bwseg{height:100%;transition:width .5s;}
.bwlgd{display:flex;gap:6px;flex-wrap:wrap;font-size:.50rem;flex-shrink:0;}
.bwlgd span{display:flex;align-items:center;gap:2px;}
.bwdot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
/* Profile badge */
#profBadge{
  border:1px solid var(--txtm);padding:4px 6px;font-size:.60rem;letter-spacing:.06em;
  text-align:center;flex-shrink:0;
}
#profBadge.pg{border-color:var(--grnd);color:var(--grn);}
#profBadge.pa{border-color:var(--ambd);color:var(--amb);}
#profBadge.pr{border-color:var(--red);color:var(--red);}
/* Frame loss bar */
.losstrack{height:8px;background:var(--bgpa);border:1px solid var(--bdr);border-radius:1px;overflow:hidden;flex-shrink:0;}
.lossfill{height:100%;transition:width .5s;}
/* Telemetry sparklines still used in network chart */
canvas{display:block;}
/* Toast */
#toast{
  position:fixed;bottom:40px;left:50%;transform:translateX(-50%);
  background:#1a2a1a;border:1px solid var(--grnd);color:var(--txt);
  font-size:.65rem;padding:5px 14px;letter-spacing:.06em;z-index:99999;
  opacity:0;transition:opacity .2s;pointer-events:none;white-space:nowrap;
}
#toast.vis{opacity:1;}
/* Offline dimming */
body.offline .panel .pb{opacity:.3;}
body.offline #offlineBanner{display:flex;}
body.degraded #degradedBar{display:flex;}
</style>
</head>
<body>

<!-- Status bar -->
<div id="sbar">
  <span style="color:var(--grn);font-weight:700;letter-spacing:.13em">FENRIR-1 MISSION CONTROL</span>
  <span class="sep">|</span><span>JETSON AGX ORIN</span>
  <span class="sep">|</span><span id="sclock">----</span>
  <span class="sep">|</span><span>GPU: <span id="sGpu">--.-</span>&deg;C [<span id="sGpuLbl">--</span>]</span>
  <span class="sep">|</span><span id="sStreams" style="color:var(--txtd)">● 0 STREAMS ACTIVE</span>
  <span class="sep">|</span><span id="sNetSim" style="color:var(--txtd)">SIM: NONE</span>
</div>

<!-- Offline banner -->
<div id="offlineBanner">
  <span class="blink2">&#9888;</span>
  &nbsp;&nbsp;ALL STREAMS OFFLINE — JETSON PROCESSES NOT RUNNING&nbsp;&nbsp;
  <span class="blink2">&#9888;</span>
</div>

<!-- Degraded bar -->
<div id="degradedBar" id="degradedBar">
  <span id="degradedMsg">DEGRADED — 0/3 STREAMS ACTIVE</span>
</div>

<!-- Main -->
<div id="main">

  <!-- Top row -->
  <div id="topRow">

    <!-- A: Camera -->
    <div class="panel" id="pCam">
      <div class="ph">[ &#9672; CAMERA FEED ]</div>
      <div class="pb" style="position:relative;">
        <div id="vcont">
        <img id="vfeed" src="" alt=""> <div id="nosig">[ CAMERA OFFLINE ]</div>
        <div id="livebadge"><span class="blink">&#9632;</span> LIVE</div>
        <div id="oTL" style="position:absolute;top:6px;left:8px;font-size:0.62rem;color:var(--green-bright);background:rgba(0,0,0,0.55);padding:1px 5px;">AGE: <span id="vAge">--.-</span>s</div>
        <div id="oBL">DET: <span id="ovDet">000</span></div>
        <div id="oBR"><span id="ovFps">--.--</span> FPS</div>
      </div>
      </div>
    </div>

    <!-- B: LiDAR -->
    <div class="panel" id="pLidar">
      <div class="ph" id="lidarHdr">[ &#9672; LIDAR POLAR SWEEP ]</div>
      <div class="pb" style="align-items:center;justify-content:center;padding:4px;">
        <canvas id="lidarC"></canvas>
      </div>
    </div>

    <!-- G: Sim View (hidden by default) -->
    <div class="panel" id="pSim">
      <div class="ph">[ &#9672; SIM VIEW — FENRIR-1 ]</div>
      <div class="pb" style="position:relative;">
        <div id="simFeedWrap">
          <img id="simFeed" alt="">
          <div id="simOffline">
            <span style="color:var(--txtm)">&#9632;</span>
            <span>SIMULATOR OFFLINE</span>
          </div>
        </div>
      </div>
    </div>

    <!-- C: Controls -->
    <div class="panel" id="pCtrl">
      <div class="ph">[ &#9672; OPERATOR CONTROLS ]</div>
      <div class="pb">

        <div class="csec">
          <div class="clbl">CAMERA STREAM</div>
          <div class="crow">
            <span><span class="sdot d" id="camDot">&#9679;</span> <span id="camSt">CHECKING</span></span>
            <button class="cbtn stp" id="camBtn" onclick="camToggle()">&#9632; STOP</button>
          </div>
        </div>

        <div class="csec">
          <div class="clbl">LIDAR SENSOR</div>
          <div class="crow">
            <span><span class="sdot d" id="ldrDot">&#9679;</span> <span id="ldrSt">CHECKING</span></span>
            <button class="cbtn stp" id="ldrBtn" onclick="ldrToggle()">&#9632; STOP</button>
          </div>
        </div>

        <div class="csec">
          <div class="clbl">AI STREAM</div>
          <div class="crow">
            <span><span class="sdot g" id="aiDot">&#9679;</span> <span id="aiSt">RUNNING</span></span>
            <button class="cbtn stp" id="aiBtn" onclick="aiToggle()">&#9632; STOP</button>
          </div>
        </div>

        <div class="csec">
          <div class="clbl">DATA LOGGING</div>
          <div class="crow">
            <span><span class="sdot d" id="logDot">&#9675;</span> <span id="logSt">INACTIVE</span></span>
            <button class="cbtn" id="logBtn" onclick="logToggle()">&#9654; START</button>
          </div>
        </div>

        <div class="csec">
          <div class="clbl">DISPLAY</div>
          <div class="crow">
            <span style="color:var(--txtd);font-size:.58rem;">SIM VIEW</span>
            <button class="cbtn" id="simTogBtn" onclick="toggleSim()">&#9654; SHOW SIM</button>
          </div>
        </div>

        <div class="csec">
          <div class="clbl">NETWORK SIMULATION</div>
          <div style="color:var(--txtd);font-size:.55rem;margin-bottom:3px;">
            ACTIVE: <span id="netSimActive" style="color:var(--grn)">NONE</span>
          </div>
          <div class="pgrid">
            <button class="pbtn" id="pb_baseline"     onclick="netStart('baseline')">BASELINE</button>
            <button class="pbtn" id="pb_good_5g"      onclick="netStart('good_5g')">GOOD 5G</button>
            <button class="pbtn" id="pb_congested_5g" onclick="netStart('congested_5g')">CONG. 5G</button>
            <button class="pbtn" id="pb_poor_5g"      onclick="netStart('poor_5g')">POOR 5G</button>
            <button class="pbtn" id="pb_satellite_ntn" onclick="netStart('satellite_ntn')">SAT. NTN</button>
            <button class="pbtn" id="pb_degraded_ntn" onclick="netStart('degraded_ntn')">DEG. NTN</button>
          </div>
          <button class="cbtn stp" style="width:100%;margin-top:3px;" onclick="netStop()">&#9632; STOP SIMULATION</button>
        </div>

        <div class="csec">
          <div class="clbl">ROBOT CONTROL</div>
          <div class="crow" style="margin-bottom:3px;">
            <span style="color:var(--txtd);font-size:.55rem;">TARGET:</span>
            <span>
              <button class="tgt sel" id="tgtSim" onclick="setTarget('sim')">SIM</button>
              <button class="tgt" id="tgtBot" onclick="setTarget('robot')">ROBOT</button>
            </span>
          </div>
          <div class="crow" style="gap:3px;margin-bottom:3px;">
            <button class="go2btn" id="btnStand" onclick="go2cmd('stand')">&#9650; STAND</button>
            <button class="go2btn" id="btnCrouch" onclick="go2cmd('crouch')">&#9660; CROUCH</button>
          </div>
          <div style="color:var(--txtd);font-size:.52rem;margin-bottom:2px;">LOCOMOTION (UNIMPL.)</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;">
            <button class="go2btn unimpl" onclick="unimplToast()">&#8593; FWD</button>
            <button class="go2btn unimpl" onclick="unimplToast()">&#8595; BACK</button>
            <button class="go2btn unimpl" onclick="unimplToast()">&#8592; LEFT</button>
            <button class="go2btn unimpl" onclick="unimplToast()">&#8594; RIGHT</button>
            <button class="go2btn unimpl" onclick="unimplToast()">&#8634; L</button>
            <button class="go2btn unimpl" onclick="unimplToast()">&#8635; R</button>
          </div>
        </div>

        <div class="csec">
          <div class="clbl">SYSTEM</div>
          <div class="crow">
            <span><span class="sdot g">&#9679;</span> yolo-saad</span>
            <span style="color:var(--txtd);font-size:.57rem;">ACTIVE</span>
          </div>
          <div style="color:var(--txtd);font-size:.57rem;margin-top:2px;">
            &#8593; <span id="uptime">00:00:00</span> uptime
          </div>
        </div>

        <button id="eStop" onclick="eStop()">&#9888; EMERGENCY STOP</button>
      </div>
    </div>

  </div><!-- /topRow -->

  <!-- Bottom row -->
  <div id="botRow">

    <!-- D: Detections -->
    <div class="panel" id="pDet">
      <div class="ph">[ &#9672; DETECTIONS ]</div>
      <div class="pb">
        <div class="dstat"><span class="lbl">COUNT:&nbsp;</span><span class="val" id="dCount">000</span></div>
        <div class="dstat"><span class="lbl">FRAME:&nbsp;</span><span class="val" id="dFrame">00000</span></div>
        <div id="dList"></div>
        <div class="sdiv">&#8212;&#8212; SESSION TOTALS &#8212;&#8212;</div>
        <div id="sList"></div>
      </div>
    </div>

    <!-- H: AI Situational Awareness -->
    <div class="panel" id="pAiSa">
      <div class="ph">[ &#9672; AI SITUATIONAL AWARENESS ]</div>
      <div class="pb">
        <div id="aiWrap"><canvas id="aiCanvas"></canvas><div id="aiOffline">AI STREAM OFFLINE</div></div>
        <div id="aiTicker"></div>
        <div id="aiRate">
          <div class="aiBar">
            <div class="aiBarLbl"><span>VIDEO</span><span id="aiVidFps">0.0 fps</span></div>
            <div class="aiBarTrack"><div class="aiBarFill" id="aiVidBar" style="width:0%;background:var(--red);"></div></div>
          </div>
          <div class="aiBar">
            <div class="aiBarLbl"><span>AI META</span><span id="aiMetFps">0.0 fps</span></div>
            <div class="aiBarTrack"><div class="aiBarFill" id="aiMetBar" style="width:0%;background:var(--grn);"></div></div>
          </div>
        </div>
      </div>
    </div>

    <!-- F: Network Analytics -->
    <div class="panel" id="pNet">
      <div class="ph">[ &#9672; NETWORK ANALYTICS ]</div>
      <div class="pb" style="flex-direction:row;align-items:stretch;padding:0;">

        <!-- F1: Bandwidth -->
        <div class="nsp" style="flex:1.5;">
          <div class="nlbl">BANDWIDTH</div>
          <div class="bwtrack">
            <div class="bwseg" id="bwVidSeg" style="background:#00aa55;width:80%;"></div>
            <div class="bwseg" id="bwLidSeg" style="background:#aa6600;width:15%;"></div>
            <div class="bwseg" id="bwMetSeg" style="background:#006699;width:5%;"></div>
          </div>
          <div class="bwlgd">
            <span><div class="bwdot" style="background:#00aa55;"></div><span id="bwVidLbl" style="color:var(--grnd)">0 KB/s VID</span></span>
            <span><div class="bwdot" style="background:#aa6600;"></div><span id="bwLidLbl" style="color:var(--ambd)">0 KB/s LID</span></span>
            <span><div class="bwdot" style="background:#006699;"></div><span id="bwMetLbl" style="color:#006699">0 KB/s META</span></span>
          </div>
          <div style="font-size:.55rem;color:var(--txtd);margin-top:2px;">
            TOTAL: <span id="bwTotal" style="color:var(--txt)">0 KB/s</span>
          </div>
          <div style="font-size:.52rem;color:var(--txtd);margin-top:4px;">CONFIGURED:</div>
          <div id="configuredProfile" style="font-size:.60rem;color:var(--amb);letter-spacing:.06em;">NONE</div>
        </div>

        <!-- F2: 3-tier chart -->
        <div class="nsp" style="flex:2;padding:4px 5px;">
          <div class="nlbl" style="margin-bottom:2px;">THREE-TIER STREAM RATES</div>
          <canvas id="chartTier" style="flex:1;width:100%;"></canvas>
          <div style="display:flex;gap:8px;font-size:.48rem;flex-shrink:0;margin-top:2px;">
            <span style="color:#00aa55;">&#9472; VIDEO</span>
            <span style="color:#aa6600;">&#9472; LIDAR&times;3</span>
            <span style="color:#006699;">&#9472; META&divide;5</span>
          </div>
        </div>

        <!-- F3: Latency -->
        <div class="nsp" style="flex:1;align-items:center;">
          <div class="nlbl">E2E LATENCY</div>
          <canvas id="gaugeLatency" width="90" height="70" style="flex-shrink:0;"></canvas>
          <div class="nval g" id="latVal" style="font-size:.95rem;">-- ms</div>
          <div style="font-size:.50rem;color:var(--txtd);">P95: <span id="latP95">-- ms</span></div>
        </div>

        <!-- F4: Quality score -->
        <div class="nsp" style="flex:1;align-items:center;">
          <div class="nlbl">QUALITY SCORE</div>
          <canvas id="gaugeQuality" width="90" height="70" style="flex-shrink:0;"></canvas>
          <div class="nval g" id="qualVal" style="font-size:1.3rem;">--</div>
          <div style="font-size:.48rem;color:var(--txtd);">FPS×.4 | LAT×.3 | DEL×.3</div>
        </div>

        <!-- F5: Profile badge -->
        <div class="nsp" style="flex:1;align-items:center;justify-content:center;gap:5px;">
          <div class="nlbl">INFERRED LINK</div>
          <div id="profBadge" class="pg" style="width:90%;text-align:center;">[ Baseline ]</div>
          <div style="font-size:.52rem;color:var(--txtd);">CONFIDENCE</div>
          <div id="profConf" style="font-size:.85rem;color:var(--grn);">0%</div>
        </div>

        <!-- F6: Frame loss -->
        <div class="nsp" style="flex:1;align-items:center;gap:4px;">
          <div class="nlbl">FRAME LOSS</div>
          <div class="nval g" id="lossVal">0.0%</div>
          <div class="losstrack" style="width:80%;">
            <div class="lossfill" id="lossFill" style="width:0%;background:var(--grn);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;width:80%;font-size:.48rem;color:var(--txtm);">
            <span>0%</span><span>5%</span><span>10%</span>
          </div>
        </div>

      </div>
    </div>

  </div><!-- /botRow -->

</div><!-- /main -->

<!-- Toast -->
<div id="toast"></div>

<script>
// ── Global state ─────────────────────────────────────────────────────────────
window.onerror = msg => {
  const el = document.getElementById('sStreams');
  if (el) { el.textContent='JS ERR: '+msg; el.style.color='#ff3333'; }
  return false;
};

const fpsHist=[], hzHist=[];
let latestLidar={active:false,points:[],rotation_speed_deg_s:4500};
let lidarScans=[];
let sweepAngle=0, lastRaf=performance.now(), rafFrame=0;
let logActive=false, simShown=false;

// Network analytics history
const tierHistory=[];    // {t,fps,hz} for 60s
const latencyHistory=[]; // latency_ms values for last 60s with timestamps {t,v}
let currentTarget='sim';

// ── WebSocket ────────────────────────────────────────────────────────────────
let ws;
function connectWS() {
  try {
    ws=new WebSocket('ws://'+window.location.host+'/ws');
    ws.onmessage=e=>{try{handleTelemetry(JSON.parse(e.data));}catch(err){}};
    ws.onclose=()=>setTimeout(connectWS,2000);
    ws.onerror=()=>{try{ws.close();}catch(e){}};
  } catch(e){setTimeout(connectWS,3000);}
}
connectWS();

// ── Telemetry handler ────────────────────────────────────────────────────────
function handleTelemetry(d) {
  if (d.type!=='telemetry') return;
  const cam=d.camera, det=d.detections, lidar=d.lidar, sys=d.system, st=d.streams, net=d.network;

  // System state / offline banner
  document.body.classList.toggle('offline', d.system_state==='offline');
  document.body.classList.toggle('degraded', d.system_state==='degraded');

  const alive=[st.video_alive,st.ai_alive,st.lidar_alive].filter(Boolean).length;
  if (d.system_state==='degraded') {
    document.getElementById('degradedMsg').textContent='DEGRADED — '+alive+'/3 STREAMS ACTIVE';
  }

  // Health dots force-red when offline
  const forceRed = d.system_state==='offline';

  // Camera panel — uses video_alive (port 8090 TCP check) independently
  const dead = !st.video_alive;
  document.getElementById('nosig').style.display = dead ? 'flex' : 'none';
  document.getElementById('vfeed').style.display = dead ? 'none' : 'block';
  document.getElementById('ovDet').textContent = String(det.count).padStart(3,'0');
  document.getElementById('ovFps').textContent = cam.fps.toFixed(1);

  // --- NEW: Calculate Video Staleness ---
  if (cam.send_timestamp) {
    const ageStr = ((Date.now() / 1000) - cam.send_timestamp).toFixed(1);
    const ageEl = document.getElementById('vAge');
    ageEl.textContent = ageStr;
    const ageVal = parseFloat(ageStr);
    
    // Color coding based on latency
    if (ageVal < 0.5) ageEl.parentElement.style.color = 'var(--green-bright)';
    else if (ageVal < 2.0) ageEl.parentElement.style.color = 'var(--amber)';
    else ageEl.parentElement.style.color = 'var(--red-alert)';
  }

  // Detections
  document.getElementById('dCount').textContent=String(det.count).padStart(3,'0');
  document.getElementById('dFrame').textContent=String(cam.frame_number).padStart(5,'0');
  renderDets(det.items);
  if (d.session_totals) renderSession(d.session_totals);

  // AI Situational Awareness panel
  updateAiSa(det, cam, d.ai_stream_enabled);

  // LiDAR
  latestLidar=lidar;
  if (lidar.active&&lidar.points.length) {
    lidarScans.push(lidar.points);
    if (lidarScans.length>3) lidarScans.shift();
  }
  document.getElementById('lidarHdr').textContent=
    '[ ◈ LIDAR | SCAN #'+lidar.scan_id+' | '+lidar.hz.toFixed(1)+' Hz | '+lidar.num_points+' pts ]';

  // Health dots
  if (forceRed) {
    ['hVid','hMeta','hLidar'].forEach(id=>document.getElementById(id).className='hdot r');
  } else {
    healthDot('hVid', st.video_alive?cam.last_seen_s:999);
    healthDot('hMeta', st.ai_alive?cam.last_seen_s:999);
    healthDot('hLidar', st.lidar_alive?lidar.last_seen_s:999);
  }

  // GPU
  const t=sys.gpu_temp_c;
  document.getElementById('tGpu').textContent=t.toFixed(1);
  document.getElementById('tGpu').className='tval '+(t>85?'r':t>70?'a':'g');
  document.getElementById('gpuBadge').textContent=sys.gpu_status.toUpperCase();
  document.getElementById('gpuBadge').className='gbadge '+sys.gpu_status;
  document.getElementById('sGpu').textContent=t.toFixed(1);
  document.getElementById('sGpuLbl').textContent=sys.gpu_status.toUpperCase();

  // Status bar
  const ss=document.getElementById('sStreams');
  ss.textContent='● '+alive+' STREAMS ACTIVE';
  ss.style.color=alive===3?'var(--grn)':alive>0?'var(--amb)':'var(--red)';

  // Network sim status bar
  const sn=document.getElementById('sNetSim');
  sn.textContent='SIM: '+(d.active_network_profile||'NONE');
  sn.style.color=d.active_network_profile?'var(--amb)':'var(--txtd)';

  // Configured profile in F1
  document.getElementById('configuredProfile').textContent=d.active_network_profile||'NONE';

  // Uptime
  const u=sys.uptime_s;
  const hh=Math.floor(u/3600),mm=Math.floor((u%3600)/60),ss2=Math.floor(u%60);
  document.getElementById('uptime').textContent=p2(hh)+':'+p2(mm)+':'+p2(ss2);

  // Control dots
  ctrlDot('camDot','camSt',st.video_alive);
  ctrlDot('ldrDot','ldrSt',st.lidar_alive);

  // Network analytics
  if (net) updateNetworkPanel(net, cam.fps, lidar.hz);
}

function p2(n){return String(n).padStart(2,'0');}

function healthDot(id,s){document.getElementById(id).className='hdot '+(s<2?'g':s<5?'a':'r');}
function ctrlDot(dotId,stId,alive){
  document.getElementById(dotId).className='sdot '+(alive?'g':'d');
  document.getElementById(stId).textContent=alive?'RUNNING':'STOPPED';
}

// ── Network analytics ────────────────────────────────────────────────────────
function updateNetworkPanel(net, fps, hz) {
  const now=Date.now()/1000;

  // F1: Bandwidth bar
  const total=net.total_bw_kbps||1;
  const vp=Math.round((net.video_bw_kbps/total)*100);
  const lp=Math.round((net.lidar_bw_kbps/total)*100);
  const mp=100-vp-lp;
  document.getElementById('bwVidSeg').style.width=vp+'%';
  document.getElementById('bwLidSeg').style.width=lp+'%';
  document.getElementById('bwMetSeg').style.width=Math.max(0,mp)+'%';
  document.getElementById('bwVidLbl').textContent=fmtKB(net.video_bw_kbps)+' VID';
  document.getElementById('bwLidLbl').textContent=fmtKB(net.lidar_bw_kbps)+' LID';
  document.getElementById('bwMetLbl').textContent=fmtKB(net.meta_bw_kbps)+' META';
  document.getElementById('bwTotal').textContent=fmtKB(net.total_bw_kbps);

  // F2: tier history
  tierHistory.push({t:now, fps:fps, hz:hz});
  const cutoff=now-60;
  while(tierHistory.length&&tierHistory[0].t<cutoff) tierHistory.shift();
  drawTierChart();

  // F3: Latency gauge
  latencyHistory.push({t:now, v:net.e2e_latency_ms});
  while(latencyHistory.length&&latencyHistory[0].t<cutoff) latencyHistory.shift();
  const p95=calcP95(latencyHistory.map(x=>x.v));
  drawLatencyGauge(net.e2e_latency_ms, p95);
  document.getElementById('latVal').textContent=Math.round(net.e2e_latency_ms)+' ms';
  document.getElementById('latVal').className='nval '+(net.e2e_latency_ms<250?'g':net.e2e_latency_ms<1000?'a':'r');
  document.getElementById('latP95').textContent=Math.round(p95)+' ms';

  // F4: Quality ring
  drawQualityRing(net.quality_score);
  document.getElementById('qualVal').textContent=net.quality_score;
  document.getElementById('qualVal').className='nval '+(net.quality_score>=80?'g':net.quality_score>=60?'a':'r');

  // F5: Profile badge
  const badge=document.getElementById('profBadge');
  badge.textContent='[ '+net.inferred_profile+' ]';
  const ntnProfiles=['Satellite NTN','Degraded NTN'];
  const ambProfiles=['Congested 5G','Poor 5G'];
  if (ntnProfiles.includes(net.inferred_profile)) badge.className='pr';
  else if (ambProfiles.includes(net.inferred_profile)) badge.className='pa';
  else badge.className='pg';
  document.getElementById('profConf').textContent=Math.round(net.profile_confidence*100)+'%';
  document.getElementById('profConf').style.color=
    net.profile_confidence>0.7?'var(--grn)':net.profile_confidence>0.4?'var(--amb)':'var(--red)';

  // F6: Frame loss
  document.getElementById('lossVal').textContent=net.frame_loss_pct.toFixed(1)+'%';
  document.getElementById('lossVal').className='nval '+(net.frame_loss_pct<1?'g':net.frame_loss_pct<5?'a':'r');
  const lossPct=Math.min(100,(net.frame_loss_pct/10)*100);
  const lf=document.getElementById('lossFill');
  lf.style.width=lossPct+'%';
  lf.style.background=net.frame_loss_pct<1?'var(--grn)':net.frame_loss_pct<5?'var(--amb)':'var(--red)';
}

function fmtKB(v){
  if(v>=1000) return (v/1000).toFixed(1)+' MB/s';
  return Math.round(v)+' KB/s';
}

function calcP95(arr){
  if(!arr.length) return 0;
  const sorted=[...arr].sort((a,b)=>a-b);
  return sorted[Math.floor(sorted.length*0.95)];
}

// F2: Three-tier chart
function drawTierChart(){
  const c=document.getElementById('chartTier');
  if(!c) return;
  const pb=c.parentElement;
  const W=pb.clientWidth-10, H=pb.clientHeight-28;
  if(W<20||H<20) return;
  if(c.width!==W||c.height!==H){c.width=W;c.height=H;}
  const ctx=c.getContext('2d');
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#040806';
  ctx.fillRect(0,0,W,H);
  if(tierHistory.length<2) return;

  const maxY=30;
  // Threshold line at 15fps
  const ty=H-(15/maxY)*H;
  ctx.strokeStyle='rgba(255,51,51,0.4)';ctx.setLineDash([4,4]);ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,ty);ctx.lineTo(W,ty);ctx.stroke();
  ctx.setLineDash([]);

  const span=60;
  const now=Date.now()/1000;

  function drawLine(key,scale,color){
    ctx.strokeStyle=color;ctx.lineWidth=1.5;
    ctx.beginPath();
    let first=true;
    tierHistory.forEach(p=>{
      const x=((p.t-(now-span))/span)*W;
      const v=Math.min((p[key]||0)*scale,maxY);
      const y=H-(v/maxY)*H;
      if(first){ctx.moveTo(x,y);first=false;}
      else ctx.lineTo(x,y);
    });
    ctx.stroke();
  }

  drawLine('fps',1,'#00aa55');
  drawLine('hz',3,'#aa6600');
  // Meta FPS ≈ same as camera fps but divide by 5 for display
  ctx.strokeStyle='#006699';ctx.lineWidth=1.5;
  ctx.beginPath();
  let first=true;
  tierHistory.forEach(p=>{
    const x=((p.t-(now-span))/span)*W;
    const v=Math.min((p.fps||0)/5,maxY);
    const y=H-(v/maxY)*H;
    if(first){ctx.moveTo(x,y);first=false;}
    else ctx.lineTo(x,y);
  });
  ctx.stroke();

  // Y axis label
  ctx.fillStyle='var(--txtm)';ctx.font='8px monospace';ctx.textAlign='left';
  ctx.fillText('30',2,10);ctx.fillText('0',2,H-2);
}

// F3: Latency gauge (arc)
function drawLatencyGauge(val, p95){
  const c=document.getElementById('gaugeLatency');
  if(!c) return;
  const ctx=c.getContext('2d');
  const W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  const cx=W/2,cy=H*0.75,R=Math.min(W,H)*0.45;
  const startA=Math.PI,endA=2*Math.PI;

  // Background arc
  ctx.beginPath();ctx.arc(cx,cy,R,startA,endA);
  ctx.strokeStyle='#1a2a1a';ctx.lineWidth=8;ctx.stroke();

  // Color segments
  const maxVal=2500;
  function arcSeg(from,to,color){
    const a1=startA+(from/maxVal)*Math.PI;
    const a2=startA+(to/maxVal)*Math.PI;
    ctx.beginPath();ctx.arc(cx,cy,R,a1,a2);
    ctx.strokeStyle=color;ctx.lineWidth=8;ctx.stroke();
  }
  arcSeg(0,250,'#00aa55');
  arcSeg(250,1000,'#aa6600');
  arcSeg(1000,2500,'#aa2200');

  // Needle
  const clampedVal=Math.min(val,maxVal);
  const needleA=startA+(clampedVal/maxVal)*Math.PI;
  ctx.save();ctx.translate(cx,cy);ctx.rotate(needleA);
  ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(R*0.85,0);
  ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.stroke();
  ctx.restore();

  // P95 tick
  if(p95>0){
    const p95A=startA+(Math.min(p95,maxVal)/maxVal)*Math.PI;
    ctx.save();ctx.translate(cx,cy);ctx.rotate(p95A);
    ctx.beginPath();ctx.moveTo(R*0.6,0);ctx.lineTo(R*0.95,0);
    ctx.strokeStyle='rgba(255,170,0,0.7)';ctx.lineWidth=1.5;ctx.stroke();
    ctx.restore();
  }
}

// F4: Quality ring
function drawQualityRing(score){
  const c=document.getElementById('gaugeQuality');
  if(!c) return;
  const ctx=c.getContext('2d');
  const W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  const cx=W/2,cy=H*0.7,R=Math.min(W,H)*0.45;
  ctx.beginPath();ctx.arc(cx,cy,R,Math.PI,2*Math.PI);
  ctx.strokeStyle='#1a2a1a';ctx.lineWidth=9;ctx.stroke();
  const fillA=Math.PI+(score/100)*Math.PI;
  const color=score>=80?'#00aa55':score>=60?'#aa6600':'#aa2200';
  ctx.beginPath();ctx.arc(cx,cy,R,Math.PI,fillA);
  ctx.strokeStyle=color;ctx.lineWidth=9;ctx.stroke();
}

// ── LiDAR canvas ─────────────────────────────────────────────────────────────
const lidarC=document.getElementById('lidarC');
const lidarX=lidarC?lidarC.getContext('2d'):null;
const MAX_MM=5000;

function sizeLidar(){
  if(!lidarC) return;
  const pb=document.getElementById('pLidar').querySelector('.pb');
  const sz=Math.max(Math.min(pb.clientWidth-8,pb.clientHeight-8),80);
  if(sz!==lidarC.width){lidarC.width=sz;lidarC.height=sz;}
}
if(window.ResizeObserver){
  new ResizeObserver(sizeLidar).observe(document.getElementById('pLidar'));
} else {
  window.addEventListener('resize',sizeLidar);
}
setTimeout(sizeLidar,0);

function drawLidar(sweep){
  if(!lidarX) return;
  const W=lidarC.width,H=lidarC.height;
  if(W<4||H<4) return;
  const cx=W/2,cy=H/2,R=Math.min(W,H)/2-18;
  lidarX.clearRect(0,0,W,H);
  lidarX.fillStyle='#080c0a';lidarX.fillRect(0,0,W,H);

  lidarX.strokeStyle='#1a3020';lidarX.setLineDash([4,4]);lidarX.lineWidth=1;
  lidarX.fillStyle='#2a4530';lidarX.font='8px monospace';
  for(let i=1;i<=4;i++){
    const rr=R*(i/4);
    lidarX.beginPath();lidarX.arc(cx,cy,rr,0,Math.PI*2);lidarX.stroke();
    lidarX.textAlign='left';
    lidarX.fillText((MAX_MM/4*i/1000).toFixed(1)+'m',cx+rr*0.71+2,cy-rr*0.71-2);
  }
  lidarX.setLineDash([]);

  lidarX.beginPath();lidarX.moveTo(cx,cy);
  lidarX.arc(cx,cy,R,260*Math.PI/180,280*Math.PI/180,false);
  lidarX.closePath();
  lidarX.fillStyle='rgba(0,255,136,.04)';lidarX.fill();
  lidarX.strokeStyle='rgba(0,255,136,.18)';lidarX.lineWidth=1;lidarX.stroke();

  lidarX.fillStyle='#3a6045';lidarX.font='9px monospace';
  [['FWD',0],['R',90],['AFT',180],['L',270]].forEach(([lbl,deg])=>{
    const rad=(deg-90)*Math.PI/180;
    lidarX.textAlign='center';
    lidarX.fillText(lbl,cx+(R+13)*Math.cos(rad),cy+(R+13)*Math.sin(rad)+3);
  });

  const ops=[0.15,0.3,1.0];
  lidarScans.forEach((pts,si)=>{
    const op=ops[si+(3-lidarScans.length)];
    const buckets=Array.from({length:10},()=>[]);
    pts.forEach(([ang,dist,conf])=>{
      if(!dist||conf<50) return;
      const dc=Math.min(dist,MAX_MM);
      const rad=(ang-90)*Math.PI/180;
      const pr=(dc/MAX_MM)*R;
      const bk=Math.min(Math.floor((dc/MAX_MM)*10),9);
      buckets[bk].push(cx+pr*Math.cos(rad),cy+pr*Math.sin(rad));
    });
    buckets.forEach((coords,bk)=>{
      if(!coords.length) return;
      const t=(bk+0.5)/10;
      let r2,g2;
      if(t<0.5){r2=Math.round(0xaa*t*2);g2=Math.round(0xff-0x55*t*2);}
      else{const f=(t-0.5)*2;r2=Math.round(0xaa+0x55*f);g2=Math.round(0xaa*(1-f));}
      lidarX.fillStyle='rgba('+r2+','+g2+',0,'+op+')';
      lidarX.beginPath();
      for(let i=0;i<coords.length;i+=2)
        lidarX.rect(coords[i]-1.5,coords[i+1]-1.5,3,3);
      lidarX.fill();
    });
  });

  if(latestLidar.active){
    const srad=(sweep-90)*Math.PI/180;
    lidarX.strokeStyle='rgba(0,255,136,.55)';lidarX.lineWidth=1.5;
    lidarX.beginPath();lidarX.moveTo(cx,cy);
    lidarX.lineTo(cx+R*Math.cos(srad),cy+R*Math.sin(srad));lidarX.stroke();
  }

  lidarX.strokeStyle='#00ff88';lidarX.lineWidth=1;
  lidarX.beginPath();
  lidarX.moveTo(cx-7,cy);lidarX.lineTo(cx+7,cy);
  lidarX.moveTo(cx,cy-7);lidarX.lineTo(cx,cy+7);
  lidarX.stroke();
  lidarX.fillStyle='#00aa55';lidarX.font='8px monospace';lidarX.textAlign='center';
  lidarX.fillText('[G2]',cx,cy+18);

  if(!latestLidar.active){
    lidarX.fillStyle='rgba(0,0,0,.65)';lidarX.fillRect(0,0,W,H);
    lidarX.fillStyle='#ffaa00';lidarX.font='11px monospace';lidarX.textAlign='center';
    lidarX.fillText('[ NO SIGNAL ]',cx,cy);
  }
}

// RAF loop
function rafLoop(ts){
  try{
    const dt=Math.min((ts-lastRaf)/1000,.1);
    lastRaf=ts;
    sweepAngle=(sweepAngle+(latestLidar.rotation_speed_deg_s||4500)*dt)%360;
    rafFrame++;
    if(rafFrame%4===0) drawLidar(sweepAngle);
  }catch(e){}
  requestAnimationFrame(rafLoop);
}
requestAnimationFrame(rafLoop);

// ── Missing telemetry strip elements (kept for structure) ─────────────────────
// These divs are referenced by id in some paths; provide stubs if not in DOM
function gauge(id,val,thresholds,invert){
  const el=document.getElementById(id);
  if(!el) return;
  el.textContent=val.toFixed(1);
  const [hi,lo]=thresholds;
  let cls='g';
  if(!invert){if(val<lo)cls='r';else if(val<hi)cls='a';}
  else{if(val>lo)cls='r';else if(val>hi)cls='a';}
  el.className='tval '+cls;
}

// ── Detections ────────────────────────────────────────────────────────────────
function renderDets(items){
  const sorted=[...items].sort((a,b)=>b.confidence-a.confidence).slice(0,8);
  document.getElementById('dList').innerHTML=sorted.map(d=>{
    const b=Math.round(d.confidence*5);
    const bar='█'.repeat(b)+'░'.repeat(5-b);
    const col=d.confidence>0.7?'var(--grn)':d.confidence>0.4?'var(--amb)':'var(--red)';
    return '<div class="ditem"><span class="dcls">'+d.class.substring(0,9)+'</span>'
      +'<span class="dbar" style="color:'+col+'">'+bar+'</span>'
      +'<span class="dconf">'+d.confidence.toFixed(2)+'</span></div>';
  }).join('');
}

function renderSession(totals){
  const sorted=Object.entries(totals).sort((a,b)=>b[1]-a[1]).slice(0,8);
  document.getElementById('sList').innerHTML=sorted.map(([cls,cnt])=>
    '<div class="sitem"><span>'+cls+':</span><span style="color:var(--grnd)">'+
    String(cnt).padStart(6)+'</span></div>'
  ).join('');
}

// ── Video signal ─────────────────────────────────────────────────────────────
const vfeed = document.getElementById('vfeed');

// Always use the dashboard's /video endpoint. The server routes internally to
// port 8090 (direct) or port 9090 (ConstrainedHTTPProxy) based on sim state,
// so this URL is reachable regardless of where the browser is running.
vfeed.src = '/video';

let videoRetryTimer = null;
function scheduleVideoRetry() {
  if (videoRetryTimer) return;
  videoRetryTimer = setTimeout(() => {
    videoRetryTimer = null;
    vfeed.src = '/video?t=' + Date.now();
  }, 5000);
}
vfeed.onerror=()=>{
  document.getElementById('nosig').style.display='flex';
  vfeed.style.display='none';
  scheduleVideoRetry();
};
vfeed.onload=()=>{
  if(videoRetryTimer){clearTimeout(videoRetryTimer);videoRetryTimer=null;}
  document.getElementById('nosig').style.display='none';
  vfeed.style.display='block';
};

// ── Sim view ──────────────────────────────────────────────────────────────────
function toggleSim(){
  simShown=!simShown;
  const pSim=document.getElementById('pSim');
  const btn=document.getElementById('simTogBtn');
  pSim.style.display=simShown?'flex':'none';
  btn.textContent=simShown?'■ HIDE SIM':'▶ SHOW SIM';

  if(simShown){
    const host=window.location.hostname;
    const sf=document.getElementById('simFeed');
    sf.src='http://'+host+':8093/sim-stream';
    sf.onerror=()=>{
      document.getElementById('simOffline').style.display='flex';
      sf.style.display='none';
    };
    sf.onload=()=>{
      document.getElementById('simOffline').style.display='none';
      sf.style.display='block';
    };
  }
}

// ── Network simulation ────────────────────────────────────────────────────────
let activeNetProfile=null;
const profileLabels={
  baseline:'Baseline', good_5g:'Good 5G', congested_5g:'Congested 5G',
  poor_5g:'Poor 5G', satellite_ntn:'Satellite NTN', degraded_ntn:'Degraded NTN'
};

function netStart(profile){
  fetch('/control/network/start/'+profile,{method:'POST'})
    .then(r=>r.json())
    .then(data=>{
      if(data.ok){
        activeNetProfile=profile;
        updateNetSimUI();
        showToast('SIM ACTIVE: '+profileLabels[profile]);
        // Reconnect /video — server now picks port 9090 (constrained proxy)
        vfeed.src='/video?t='+Date.now();
      }
    })
    .catch(()=>showToast('Network sim error'));
}

function netStop(){
  fetch('/control/network/stop',{method:'POST'})
    .then(r=>r.json())
    .then(data=>{
      if(data.ok){
        activeNetProfile=null;
        updateNetSimUI();
        showToast('SIMULATION STOPPED');
        // Reconnect /video — server now picks port 8090 (direct)
        vfeed.src='/video?t='+Date.now();
      }
    })
    .catch(()=>showToast('Network sim error'));
}

function updateNetSimUI(){
  const ntnProfiles=['satellite_ntn','degraded_ntn'];
  document.getElementById('netSimActive').textContent=
    activeNetProfile?profileLabels[activeNetProfile]:'NONE';
  document.getElementById('netSimActive').style.color=
    !activeNetProfile?'var(--grn)':ntnProfiles.includes(activeNetProfile)?'var(--red)':'var(--amb)';
  // Update profile buttons
  Object.keys(profileLabels).forEach(k=>{
    const btn=document.getElementById('pb_'+k);
    if(!btn) return;
    btn.className='pbtn'+(k===activeNetProfile?(ntnProfiles.includes(k)?' pact-ntn':' pact'):'');
  });
}

// ── Robot control ─────────────────────────────────────────────────────────────
function go2cmd(cmd){
  const btn=document.getElementById('btn'+cmd.charAt(0).toUpperCase()+cmd.slice(1));
  fetch('/go2/'+cmd,{method:'POST'})
    .then(r=>r.json())
    .then(data=>{
      if(data.success){
        showToast('✓ '+cmd.toUpperCase()+' SENT');
        if(btn){btn.classList.add('flash');setTimeout(()=>btn.classList.remove('flash'),500);}
      } else {
        showToast(data.message||'Command failed');
      }
    })
    .catch(()=>showToast('GO2 API OFFLINE'));
}

function unimplToast(){
  showToast('NOT IMPLEMENTED — Requires onboard gait controller');
}

function setTarget(t){
  fetch('/go2-target/'+t,{method:'POST'})
    .then(r=>r.json())
    .then(data=>{
      if(data.ok){
        currentTarget=t;
        document.getElementById('tgtSim').className='tgt'+(t==='sim'?' sel':'');
        document.getElementById('tgtBot').className='tgt'+(t==='robot'?' sel':'');
        showToast('TARGET: '+t.toUpperCase());
      }
    })
    .catch(()=>showToast('Target switch error'));
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer=null;
function showToast(msg){
  const el=document.getElementById('toast');
  el.textContent=msg;el.classList.add('vis');
  if(toastTimer) clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>{el.classList.remove('vis');},2500);
}

// ── Controls ──────────────────────────────────────────────────────────────────
function camToggle(){
  const running=document.getElementById('camSt').textContent==='RUNNING';
  fetch('/control/camera/'+(running?'stop':'start'),{method:'POST'});
}
function ldrToggle(){
  const running=document.getElementById('ldrSt').textContent==='RUNNING';
  fetch('/control/lidar/'+(running?'stop':'start'),{method:'POST'});
}
function logToggle(){
  logActive=!logActive;
  fetch('/control/logging/'+(logActive?'start':'stop'),{method:'POST'});
  document.getElementById('logDot').className='sdot '+(logActive?'g':'d');
  document.getElementById('logSt').textContent=logActive?'ACTIVE':'INACTIVE';
  document.getElementById('logBtn').textContent=logActive?'■ STOP':'▶ START';
  document.getElementById('logBtn').className='cbtn'+(logActive?' stp':'');
}
let aiStreamActive=true;
function aiToggle(){
  aiStreamActive=!aiStreamActive;
  fetch('/control/aistream/'+(aiStreamActive?'start':'stop'),{method:'POST'});
  document.getElementById('aiDot').className='sdot '+(aiStreamActive?'g':'d');
  document.getElementById('aiSt').textContent=aiStreamActive?'RUNNING':'STOPPED';
  document.getElementById('aiBtn').textContent=aiStreamActive?'■ STOP':'▶ START';
  document.getElementById('aiBtn').className='cbtn'+(aiStreamActive?' stp':'');
}
function eStop(){
  Promise.all([
    fetch('/control/camera/stop',{method:'POST'}),
    fetch('/control/lidar/stop',{method:'POST'}),
    fetch('/control/logging/stop',{method:'POST'}),
    fetch('/control/network/stop',{method:'POST'}),
  ]);
  logActive=false;
  activeNetProfile=null;
  updateNetSimUI();
  // Reconnect /video — drops any active proxy connection
  vfeed.src='/video?t='+Date.now();
  document.getElementById('logDot').className='sdot d';
  document.getElementById('logSt').textContent='INACTIVE';
  document.getElementById('logBtn').textContent='▶ START';
  document.getElementById('logBtn').className='cbtn';
  showToast('EMERGENCY STOP — ALL SERVICES HALTED');
}

// Telemetry strip stubs (for healthDot calls in handleTelemetry)
// These elements exist in the network panel now; stub any missing
['hVid','hMeta','hLidar'].forEach(id=>{
  if(!document.getElementById(id)){
    const d=document.createElement('div');d.id=id;d.style.display='none';
    document.body.appendChild(d);
  }
});
['tGpu','gpuBadge'].forEach(id=>{
  if(!document.getElementById(id)){
    const d=document.createElement('span');d.id=id;d.style.display='none';
    document.body.appendChild(d);
  }
});

// ── Clock ─────────────────────────────────────────────────────────────────────
function tick(){
  const iso=new Date().toISOString();
  document.getElementById('sclock').textContent=iso.slice(0,10)+'  '+iso.slice(11,19)+' UTC';
}
tick();setInterval(tick,1000);

// ── Resize tier chart on panel resize ────────────────────────────────────────
if(window.ResizeObserver){
  new ResizeObserver(drawTierChart).observe(document.getElementById('pNet'));
}

// ── AI Situational Awareness ─────────────────────────────────────────────────
const aiCanvas = document.getElementById('aiCanvas');
const aiCtx = aiCanvas.getContext('2d');
const aiTicker = document.getElementById('aiTicker');
const aiTickerItems = [];
let aiLastTs = 0;

function resizeAiCanvas() {
  const wrap = document.getElementById('aiWrap');
  if (!wrap) return;
  aiCanvas.width = wrap.clientWidth;
  aiCanvas.height = wrap.clientHeight;
}
resizeAiCanvas();
if (window.ResizeObserver) {
  new ResizeObserver(resizeAiCanvas).observe(document.getElementById('aiWrap'));
}

function updateAiSa(det, cam, aiEnabled) {
  const disabled = aiEnabled === false;
  const offline = !disabled && cam.last_seen_s > 2;
  const aiOff = document.getElementById('aiOffline');
  if (disabled) {
    aiOff.textContent = 'AI STREAM DISABLED';
    aiOff.style.display = 'flex';
    aiCanvas.style.display = 'none';
  } else if (offline) {
    aiOff.textContent = 'AI STREAM OFFLINE';
    aiOff.style.display = 'flex';
    aiCanvas.style.display = 'none';
  } else {
    aiOff.style.display = 'none';
    aiCanvas.style.display = 'block';
  }
  // Sync toggle UI from server state
  if (disabled !== !aiStreamActive) {
    aiStreamActive = !disabled;
    document.getElementById('aiDot').className = 'sdot ' + (aiStreamActive ? 'g' : 'd');
    document.getElementById('aiSt').textContent = aiStreamActive ? 'RUNNING' : 'STOPPED';
    document.getElementById('aiBtn').textContent = aiStreamActive ? '■ STOP' : '▶ START';
    document.getElementById('aiBtn').className = 'cbtn' + (aiStreamActive ? ' stp' : '');
  }

  // Draw bounding boxes
  const ctx = aiCtx;
  const cw = aiCanvas.width, ch = aiCanvas.height;
  if (cw === 0 || ch === 0) return;
  ctx.clearRect(0, 0, cw, ch);

  const fw = cam.frame_w || 640, fh = cam.frame_h || 480;
  const sx = cw / fw, sy = ch / fh;

  for (const d of (det.items || [])) {
    if (!d.bbox) continue;
    const [x1, y1, x2, y2] = d.bbox;
    const conf = d.confidence;
    const col = conf > 0.8 ? '#00ff88' : conf > 0.5 ? '#ffaa00' : '#ff3333';

    const rx = x1 * sx, ry = y1 * sy, rw = (x2 - x1) * sx, rh = (y2 - y1) * sy;
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.strokeRect(rx, ry, rw, rh);

    // Label background
    const label = d.class + ' ' + Math.round(conf * 100) + '%';
    ctx.font = '11px "JetBrains Mono", monospace';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(rx, ry - 15, tw + 6, 15);
    ctx.fillStyle = col;
    ctx.fillText(label, rx + 3, ry - 3);
  }

  // Ticker
  if (det.count > 0) {
    const now = Date.now();
    const delta = aiLastTs ? ((now - aiLastTs) / 1000).toFixed(2) : '0.00';
    aiLastTs = now;
    const descs = (det.items || []).map(d => {
      const col = d.confidence > 0.8 ? '#00ff88' : d.confidence > 0.5 ? '#ffaa00' : '#ff3333';
      return '<span style="color:' + col + '">' + d.class + ' (' + Math.round(d.confidence * 100) + '%)</span>';
    }).join(', ');
    const entry = '<div class="aiTick"><span class="ts">[+' + delta + 's]</span> '
      + '<span class="cnt">' + det.count + ' obj</span>: ' + descs + '</div>';
    aiTickerItems.unshift(entry);
    if (aiTickerItems.length > 8) aiTickerItems.length = 8;
    aiTicker.innerHTML = aiTickerItems.join('');
  }

  // Stream rate comparison bars
  const vFps = cam.fps || 0;
  const mFps = cam.meta_fps || 0;
  const vPct = Math.min(100, (vFps / 30) * 100);
  const mPct = Math.min(100, (mFps / 30) * 100);
  document.getElementById('aiVidFps').textContent = vFps.toFixed(1) + ' fps';
  document.getElementById('aiMetFps').textContent = mFps.toFixed(1) + ' fps';
  const vBar = document.getElementById('aiVidBar');
  const mBar = document.getElementById('aiMetBar');
  vBar.style.width = vPct + '%';
  mBar.style.width = mPct + '%';
  vBar.style.background = vPct < 30 ? 'var(--red)' : vPct < 60 ? 'var(--amb)' : 'var(--grn)';
  mBar.style.background = mPct < 30 ? 'var(--red)' : mPct < 60 ? 'var(--amb)' : 'var(--grn)';
}
</script>

<!-- Health dot elements (needed by telemetry handler) -->
<div style="display:none">
  <div id="hVid" class="hdot r"></div>
  <div id="hMeta" class="hdot r"></div>
  <div id="hLidar" class="hdot r"></div>
  <span id="tGpu">0</span>
  <span id="gpuBadge" class="gbadge nominal">NOMINAL</span>
</div>

</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws="wsproto")
