#!/usr/bin/env python3
"""
YOLO inference streaming server for network degradation experiments.

Runs INSIDE the Docker container (yolo-saad). Provides:
  - HTTP MJPEG video stream on port 8090 (/stream)
  - TCP detection metadata on port 8091 (JSON lines)

Usage (inside container):
    python3 /workspace/stream_server.py
    python3 /workspace/stream_server.py --source go2stream --go2-interface eth0
    python3 /workspace/stream_server.py --source file:/workspace/test.mp4
    python3 /workspace/stream_server.py --source rtsp://192.168.1.100:8554/live
"""

import argparse
import csv
import json
import signal
import socket
import struct
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Classes relevant to outdoor / robot-dog deployment
RELEVANT_CLASSES = {"person", "dog", "cat", "car", "bicycle", "motorcycle", "truck", "bus"}

BOX_COLORS = {
    "person":     (0, 255, 0),
    "dog":        (255, 165, 0),
    "cat":        (255, 0, 255),
    "car":        (0, 0, 255),
    "bicycle":    (255, 255, 0),
    "motorcycle": (0, 255, 255),
    "truck":      (128, 0, 255),
    "bus":        (255, 128, 0),
}

# ─── Video Source Abstraction ─────────────────────────────────────────────────

def make_webcam_source(device=0):
    """USB webcam via cv2.VideoCapture. Default source."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open webcam device {device}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SRC] Webcam opened: {w}x{h}")

    def gen():
        nonlocal cap
        while not _shutdown.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Webcam read failed, retrying...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(device)
                continue
            yield True, frame
        cap.release()

    return w, h, gen()


# Go2 onboard camera via unitree_sdk2py VideoClient RPC.
# WARNING: GetImageSample() stalls intermittently with error 3104 during
# continuous streaming (unitree_sdk2 issue #116). Use --source go2stream
# for reliable continuous capture.
# --go2-interface must match the Ethernet interface connected to the Go2
# (default eth0). The Go2 must be powered on and connected via Ethernet
# before launching stream_server.py with --source go2.
def make_go2_source(interface="eth0"):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    ChannelFactoryInitialize(0, interface)
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    print(f"[SRC] Go2 VideoClient initialised on interface {interface}")

    w, h = 640, 480

    def gen():
        while not _shutdown.is_set():
            code, data = client.GetImageSample()
            if code == 0 and data:
                frame = cv2.imdecode(
                    np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield True, frame
                    continue
            yield False, None

    return w, h, gen()


def make_go2stream_source(interface="eth0"):
    """Go2 H.264 RTP multicast via GStreamer — reliable for continuous streaming."""
    pipeline = (
        f"udpsrc address=230.1.1.1 port=1720 "
        f"multicast-iface={interface} ! queue ! "
        "application/x-rtp,media=video,encoding-name=H264 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open Go2 H.264 multicast stream on {interface}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"[SRC] Go2 H.264 multicast on {interface}: {w}x{h}")

    def gen():
        nonlocal cap
        while not _shutdown.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Go2 stream read failed, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                continue
            yield True, frame
        cap.release()

    return w, h, gen()


def make_file_source(path):
    """Video file, loops on end."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video file: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SRC] File opened: {path} ({w}x{h})")

    def gen():
        nonlocal cap
        while not _shutdown.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            yield True, frame
        cap.release()

    return w, h, gen()


def make_rtsp_source(url):
    """RTSP stream, reconnects on failure."""
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[WARN] Cannot open RTSP stream: {url}, will retry...")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"[SRC] RTSP: {url} ({w}x{h})")

    def gen():
        nonlocal cap, w, h
        while not _shutdown.is_set():
            if not cap.isOpened():
                time.sleep(2)
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or w
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or h
                continue
            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue
            yield True, frame
        cap.release()

    return w, h, gen()


def open_source(source_str, go2_interface="eth0"):
    """Parse --source value and return (width, height, frame_generator)."""
    if source_str == "webcam":
        return make_webcam_source(0)
    elif source_str == "go2":
        return make_go2_source(go2_interface)
    elif source_str == "go2stream":
        return make_go2stream_source(go2_interface)
    elif source_str.startswith("file:"):
        return make_file_source(source_str[5:])
    elif source_str.startswith("rtsp:"):
        return make_rtsp_source(source_str[5:])
    else:
        sys.exit(f"[ERROR] Unknown source: {source_str}\n"
                 f"  Valid: webcam | go2 | go2stream | file:<path> | rtsp:<url>")


# ─── Shared State ────────────────────────────────────────────────────────────

# Shared state between inference thread and HTTP/TCP servers
latest_frame_lock = threading.Lock()
latest_jpeg = None          # JPEG-encoded annotated frame
latest_metadata = None      # JSON string for detection metadata
frame_event = threading.Event()


class MJPEGHandler(BaseHTTPRequestHandler):
    """Serves MJPEG stream on /stream and a simple status page on /."""

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame_event.wait(timeout=2.0)
                    with latest_frame_lock:
                        jpeg = latest_jpeg
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n".encode())
                        self.wfile.write(b"\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
            except Exception:
                pass
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>YOLO Stream</h2>"
                b"<img src='/stream' /></body></html>")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress per-request logs


def run_mjpeg_server(port):
    """Run the MJPEG HTTP server in a thread."""
    server = HTTPServer(("0.0.0.0", port), MJPEGHandler)
    server.serve_forever()


def run_metadata_server(port):
    """TCP server that sends JSON-line detection metadata to connected clients.

    Protocol: each message is a 4-byte big-endian length prefix followed by
    a UTF-8 JSON payload.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(4)
    srv.settimeout(1.0)

    clients = []
    clients_lock = threading.Lock()

    def accept_loop():
        while not _shutdown.is_set():
            try:
                conn, addr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with clients_lock:
                    clients.append(conn)
                print(f"[META] Client connected: {addr}")
            except socket.timeout:
                continue
            except OSError:
                break

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()

    while not _shutdown.is_set():
        frame_event.wait(timeout=1.0)
        with latest_frame_lock:
            meta = latest_metadata
        if meta is None:
            continue

        payload = meta.encode("utf-8")
        header = struct.pack(">I", len(payload))
        dead = []
        with clients_lock:
            for c in clients:
                try:
                    c.sendall(header + payload)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(c)
            for c in dead:
                clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    srv.close()


def read_gpu_temp():
    thermal_base = Path("/sys/devices/virtual/thermal/")
    for zone in sorted(thermal_base.glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == "gpu-thermal":
                return int((zone / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError, TypeError):
            continue
    return None


_shutdown = threading.Event()


def main():
    parser = argparse.ArgumentParser(
        description="YOLO streaming server for network experiments")
    parser.add_argument("--model", default="/workspace/yolo26n.engine",
                        help="YOLO model path (default: yolo26n.engine)")
    parser.add_argument("--source", default="webcam",
                        help="Video source: webcam | go2 | go2stream | "
                             "file:<path> | rtsp:<url> (default: webcam)")
    parser.add_argument("--go2-interface", default="eth0",
                        help="Network interface for Go2 DDS (default: eth0)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--port", type=int, default=8090,
                        help="MJPEG HTTP port (default: 8090)")
    parser.add_argument("--meta-port", type=int, default=8091,
                        help="TCP metadata port (default: 8091)")
    parser.add_argument("--jpeg-quality", type=int, default=30,
                        help="JPEG quality 1-100 (default: 30)")
    parser.add_argument("--log", default=None,
                        help="Path for server-side per-frame CSV log "
                             "(e.g. /workspace/server_log.csv)")
    args = parser.parse_args()

    # Load model
    model_path = args.model
    p = Path(model_path)
    if not p.exists() and p.suffix == ".engine":
        fallback = p.with_suffix(".pt")
        if fallback.exists():
            print(f"[WARN] Engine not found, falling back to {fallback}")
            model_path = str(fallback)
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Open video source
    print(f"Opening source: {args.source}")
    w, h, frames = open_source(args.source, args.go2_interface)
    print(f"Source ready: {w}x{h}")

    # Start servers
    print(f"Starting MJPEG server on :{args.port}")
    mjpeg_thread = threading.Thread(
        target=run_mjpeg_server, args=(args.port,), daemon=True)
    mjpeg_thread.start()

    print(f"Starting metadata server on :{args.meta_port}")
    meta_thread = threading.Thread(
        target=run_metadata_server, args=(args.meta_port,), daemon=True)
    meta_thread.start()

    def handle_signal(sig, _frame):
        _shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    global latest_jpeg, latest_metadata

    # Server-side CSV log
    log_file = None
    log_writer = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "timestamp", "frame_number", "num_detections",
            "inference_time_ms", "jpeg_size_bytes",
        ])
        print(f"Server log: {log_path}")

    frame_count = 0
    fps_smooth = 0.0
    prev_time = time.time()
    class_counts = defaultdict(int)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    print(f"\nStreaming at http://0.0.0.0:{args.port}/stream")
    print(f"Metadata on TCP :{args.meta_port}")
    print("Press Ctrl+C to stop.\n")

    for ret, frame in frames:
        if _shutdown.is_set():
            break
        if not ret:
            continue

        frame_count += 1
        now = time.time()
        dt = now - prev_time
        prev_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = (0.9 * fps_smooth + 0.1 * instant_fps
                      if fps_smooth > 0 else instant_fps)

        # Inference
        t_infer_start = time.time()
        results = model(frame, conf=args.conf, verbose=False)
        t_infer_end = time.time()
        inference_ms = (t_infer_end - t_infer_start) * 1000

        # Draw filtered detections
        annotated = frame.copy()
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            if cls_name not in RELEVANT_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            color = BOX_COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                annotated, (ix1, iy1 - th - 6), (ix1 + tw, iy1), color, -1)
            cv2.putText(annotated, label, (ix1, iy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            class_counts[cls_name] += 1
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [round(x1, 1), round(y1, 1),
                         round(x2, 1), round(y2, 1)],
            })

        # HUD overlay
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, f"Det: {len(detections)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode JPEG
        ok, jpeg_buf = cv2.imencode(".jpg", annotated, encode_params)
        if not ok:
            continue

        # Build metadata JSON
        send_ts = time.time()
        meta = json.dumps({
            "frame_number": frame_count,
            "send_timestamp": send_ts,
            "inference_time_ms": round(inference_ms, 2),
            "detections": detections,
            "fps": round(fps_smooth, 1),
            "jpeg_bytes": len(jpeg_buf),
            "frame_w": w,
            "frame_h": h,
        })

        # Publish to shared state
        with latest_frame_lock:
            latest_jpeg = jpeg_buf.tobytes()
            latest_metadata = meta
        frame_event.set()
        frame_event.clear()

        # Server-side log
        if log_writer:
            log_writer.writerow([
                f"{send_ts:.6f}",
                frame_count,
                len(detections),
                f"{inference_ms:.2f}",
                len(jpeg_buf),
            ])
            if frame_count % 30 == 0:
                log_file.flush()

        # Terminal status
        if frame_count % 30 == 0:
            gpu_temp = read_gpu_temp()
            temp_tag = f"  GPU: {gpu_temp:.0f}C" if gpu_temp else ""
            total = sum(class_counts.values())
            print(f"  Frame {frame_count:>6d}  |  FPS: {fps_smooth:>5.1f}  |  "
                  f"Det: {total}  |  Infer: {inference_ms:.1f}ms{temp_tag}    ",
                  end="\r")

    if log_file:
        log_file.close()
    print(f"\n\nStopped after {frame_count} frames.")


if __name__ == "__main__":
    main()
