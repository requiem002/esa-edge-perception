#!/usr/bin/env python3
"""
Streaming client — measures perception quality received from stream_server.py.

Runs on the HOST (outside Docker). Connects to:
  - MJPEG video stream (port 8090) via raw HTTP — tracks per-frame bytes
  - TCP detection metadata (port 8091) — JSON length-prefixed
  - TCP LIDAR scans (port 8092, optional) — JSON length-prefixed

Three separate CSVs are written:
    <base>.csv               metadata (send_ts, recv_ts, latency, detections)
    <base_stem>_video.csv    video frames (recv_ts, byte_size)
    <base_stem>_lidar.csv    lidar scans (recv_ts, byte_size, num_points)

Usage:
    python3 ~/Desktop/stream_client.py
    python3 ~/Desktop/stream_client.py --duration 60 --output measurements.csv
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
from pathlib import Path


# ── Metadata receiver (TCP, length-prefixed JSON) ────────────────────

def receive_metadata(host, port, results, stop_event):
    """Connect to the TCP metadata server and collect JSON messages."""
    sock = None
    for attempt in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))
            sock.settimeout(5.0)
            break
        except (ConnectionRefusedError, socket.timeout):
            if attempt < 9:
                time.sleep(0.5)
            else:
                print(f"[META] Could not connect to {host}:{port}")
                return

    buf = b""
    while not stop_event.is_set():
        try:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk

            while len(buf) >= 4:
                msg_len = struct.unpack(">I", buf[:4])[0]
                if len(buf) < 4 + msg_len:
                    break
                payload = buf[4:4 + msg_len].decode("utf-8", errors="ignore")
                buf = buf[4 + msg_len:]

                recv_ts = time.time()
                try:
                    meta = json.loads(payload)
                    meta["recv_timestamp"] = recv_ts
                    results.append(meta)
                except json.JSONDecodeError:
                    pass
        except socket.timeout:
            continue
        except (ConnectionResetError, BrokenPipeError, OSError):
            break

    if sock:
        sock.close()


# ── Video receiver (raw HTTP MJPEG, tracks bytes per frame) ──────────

def _read_until(sock, buf, marker):
    """Read from sock into buf until marker is present in buf."""
    while marker not in buf:
        chunk = sock.recv(8192)
        if not chunk:
            return None
        buf += chunk
    return buf


def receive_video(host, port, video_records, stop_event):
    """Parse MJPEG multipart stream and record per-frame timestamp + bytes.

    Avoids cv2.VideoCapture so we can measure the actual byte size of
    each JPEG arriving at the client (post-proxy bandwidth throttling).
    """
    while not stop_event.is_set():
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))

            req = (f"GET /stream HTTP/1.0\r\n"
                   f"Host: {host}:{port}\r\n"
                   f"User-Agent: stream_client/1.0\r\n\r\n").encode()
            sock.sendall(req)

            buf = b""
            buf = _read_until(sock, buf, b"\r\n\r\n")
            if buf is None:
                continue
            sep = buf.index(b"\r\n\r\n")
            buf = buf[sep + 4:]

            while not stop_event.is_set():
                # Locate next part boundary
                idx = buf.find(b"--frame")
                if idx < 0:
                    chunk = sock.recv(8192)
                    if not chunk:
                        break
                    buf += chunk
                    continue
                buf = buf[idx:]
                # consume "--frame\r\n"
                if b"\r\n" not in buf[7:]:
                    chunk = sock.recv(8192)
                    if not chunk:
                        break
                    buf += chunk
                    continue
                nl = buf.index(b"\r\n", 7)
                buf = buf[nl + 2:]

                # Read part headers
                buf = _read_until(sock, buf, b"\r\n\r\n")
                if buf is None:
                    break
                hdr_end = buf.index(b"\r\n\r\n")
                part_hdr = buf[:hdr_end].decode("latin1", errors="ignore")
                buf = buf[hdr_end + 4:]

                content_length = None
                for line in part_hdr.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        try:
                            content_length = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass

                if content_length is None or content_length <= 0:
                    continue

                # Read JPEG body
                while len(buf) < content_length:
                    chunk = sock.recv(min(65536, content_length - len(buf)))
                    if not chunk:
                        raise ConnectionResetError("stream ended mid-frame")
                    buf += chunk

                _jpeg = buf[:content_length]
                buf = buf[content_length:]

                video_records.append({
                    "recv_timestamp": time.time(),
                    "byte_size": content_length,
                })

                # Strip trailing \r\n if present
                while len(buf) < 2:
                    chunk = sock.recv(8192)
                    if not chunk:
                        break
                    buf += chunk
                if buf[:2] == b"\r\n":
                    buf = buf[2:]

        except (socket.timeout, ConnectionRefusedError,
                ConnectionResetError, OSError):
            time.sleep(0.5)
        finally:
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass


# ── LIDAR receiver (TCP, length-prefixed JSON) ───────────────────────

def receive_lidar(host, port, lidar_records, stop_event):
    """Connect to LIDAR scan server and record each scan."""
    sock = None
    for attempt in range(5):
        if stop_event.is_set():
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))
            sock.settimeout(5.0)
            break
        except (ConnectionRefusedError, socket.timeout, OSError):
            if attempt < 4:
                time.sleep(0.5)
            else:
                # LIDAR not available — silent skip
                return

    buf = b""
    while not stop_event.is_set():
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 4:
                msg_len = struct.unpack(">I", buf[:4])[0]
                if len(buf) < 4 + msg_len:
                    break
                payload = buf[4:4 + msg_len]
                buf = buf[4 + msg_len:]
                recv_ts = time.time()
                num_points = 0
                try:
                    obj = json.loads(payload.decode("utf-8", errors="ignore"))
                    num_points = len(obj.get("points", []))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                lidar_records.append({
                    "recv_timestamp": recv_ts,
                    "byte_size": msg_len,
                    "num_points": num_points,
                })
        except socket.timeout:
            continue
        except (ConnectionResetError, BrokenPipeError, OSError):
            break

    if sock:
        try:
            sock.close()
        except OSError:
            pass


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stream client — measure perception under network constraints")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8090,
                        help="MJPEG port (default: 8090)")
    parser.add_argument("--meta-port", type=int, default=8091,
                        help="Metadata TCP port (default: 8091)")
    parser.add_argument("--lidar-port", type=int, default=8092,
                        help="LIDAR TCP port (default: 8092, optional)")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Disable LIDAR connection attempt")
    parser.add_argument("--duration", type=int, default=60,
                        help="Measurement duration in seconds (default: 60)")
    parser.add_argument("--output", default="measurements.csv",
                        help="Output CSV path (metadata)")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_csv = out_path.with_name(out_path.stem + "_video.csv")
    lidar_csv = out_path.with_name(out_path.stem + "_lidar.csv")

    meta_results = []
    video_records = []
    lidar_records = []
    stop_event = threading.Event()

    meta_thread = threading.Thread(
        target=receive_metadata,
        args=(args.host, args.meta_port, meta_results, stop_event),
        daemon=True)
    meta_thread.start()

    video_thread = threading.Thread(
        target=receive_video,
        args=(args.host, args.port, video_records, stop_event),
        daemon=True)
    video_thread.start()

    lidar_thread = None
    if not args.no_lidar:
        lidar_thread = threading.Thread(
            target=receive_lidar,
            args=(args.host, args.lidar_port, lidar_records, stop_event),
            daemon=True)
        lidar_thread.start()

    def handle_signal(sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Collecting for {args.duration}s from {args.host}...")
    print(f"  Video: http://{args.host}:{args.port}/stream")
    print(f"  Meta:  {args.host}:{args.meta_port}")
    if not args.no_lidar:
        print(f"  Lidar: {args.host}:{args.lidar_port} (optional)")
    print()

    t_start = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - t_start
        if elapsed >= args.duration:
            break

        n_meta = len(meta_results)
        n_video = len(video_records)
        n_lidar = len(lidar_records)

        recent = [r["recv_timestamp"] for r in video_records
                  if r["recv_timestamp"] > time.time() - 2.0]
        live_video_fps = len(recent) / 2.0 if len(recent) > 1 else 0.0

        print(f"  [{elapsed:5.0f}s / {args.duration}s]  "
              f"vid:{n_video} ({live_video_fps:.1f}fps)  "
              f"meta:{n_meta}  lidar:{n_lidar}     ", end="\r")
        time.sleep(1.0)

    stop_event.set()
    meta_thread.join(timeout=3)
    video_thread.join(timeout=3)
    if lidar_thread:
        lidar_thread.join(timeout=3)

    t_end = time.time()
    actual_duration = t_end - t_start

    # ── Write metadata CSV ──
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_number", "send_timestamp", "recv_timestamp",
            "latency_ms", "inference_time_ms", "num_detections",
            "fps_at_server", "jpeg_bytes",
        ])
        for m in meta_results:
            send_ts = m.get("send_timestamp", 0)
            recv_ts = m.get("recv_timestamp", 0)
            latency_ms = (recv_ts - send_ts) * 1000 if send_ts else 0
            writer.writerow([
                m.get("frame_number", 0),
                f"{send_ts:.6f}",
                f"{recv_ts:.6f}",
                f"{latency_ms:.2f}",
                m.get("inference_time_ms", 0),
                len(m.get("detections", [])),
                m.get("fps", 0),
                m.get("jpeg_bytes", 0),
            ])

    # ── Write video CSV ──
    with open(video_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recv_timestamp", "byte_size"])
        for r in video_records:
            writer.writerow([f"{r['recv_timestamp']:.6f}", r["byte_size"]])

    # ── Write LIDAR CSV ──
    with open(lidar_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recv_timestamp", "byte_size", "num_points"])
        for r in lidar_records:
            writer.writerow([
                f"{r['recv_timestamp']:.6f}",
                r["byte_size"],
                r["num_points"],
            ])

    # ── Compute summary ──
    n_meta = len(meta_results)
    n_video = len(video_records)
    n_lidar = len(lidar_records)

    if n_meta > 0:
        latencies = [(m["recv_timestamp"] - m["send_timestamp"]) * 1000
                     for m in meta_results if m.get("send_timestamp")]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0
        max_lat = max(latencies) if latencies else 0
        min_lat = min(latencies) if latencies else 0
        infer_times = [m["inference_time_ms"] for m in meta_results
                       if m.get("inference_time_ms")]
        mean_infer = sum(infer_times) / len(infer_times) if infer_times else 0
        total_det = sum(len(m.get("detections", [])) for m in meta_results)
    else:
        mean_lat = max_lat = min_lat = mean_infer = total_det = 0

    video_fps = n_video / actual_duration if actual_duration > 0 else 0
    meta_fps = n_meta / actual_duration if actual_duration > 0 else 0
    lidar_fps = n_lidar / actual_duration if actual_duration > 0 else 0

    video_bytes = sum(r["byte_size"] for r in video_records)
    lidar_bytes = sum(r["byte_size"] for r in lidar_records)
    video_bps = video_bytes / actual_duration if actual_duration > 0 else 0
    lidar_bps = lidar_bytes / actual_duration if actual_duration > 0 else 0

    if meta_results:
        server_frames = meta_results[-1].get("frame_number", 0) - \
                        meta_results[0].get("frame_number", 0) + 1
        delivery_rate = n_meta / server_frames if server_frames > 0 else 0
    else:
        server_frames = 0
        delivery_rate = 0

    print(f"\n\n{'=' * 65}")
    print(f"  MEASUREMENT SUMMARY  ({actual_duration:.0f}s)")
    print(f"{'=' * 65}")
    print(f"  Video frames received  : {n_video}  ({video_fps:.1f} FPS, "
          f"{video_bps / 1024:.1f} KiB/s)")
    print(f"  Metadata messages      : {n_meta}  ({meta_fps:.1f} msg/s)")
    print(f"  LIDAR scans            : {n_lidar}  ({lidar_fps:.1f} Hz, "
          f"{lidar_bps / 1024:.1f} KiB/s)")
    print(f"  Server frames produced : {server_frames}")
    print(f"  Frame delivery rate    : {delivery_rate:.1%}")
    print(f"  Latency (mean/min/max) : {mean_lat:.1f} / {min_lat:.1f} / {max_lat:.1f} ms")
    print(f"  Mean inference time    : {mean_infer:.1f} ms")
    print(f"  Total detections       : {total_det}")
    print(f"  CSVs:")
    print(f"    meta : {out_path}")
    print(f"    video: {video_csv}")
    print(f"    lidar: {lidar_csv}")
    print(f"{'=' * 65}\n")

    return {
        "duration_s": round(actual_duration, 1),
        "video_frames": n_video,
        "video_fps": round(video_fps, 2),
        "video_bytes_per_sec": round(video_bps, 1),
        "meta_messages": n_meta,
        "meta_fps": round(meta_fps, 2),
        "lidar_scans": n_lidar,
        "lidar_fps": round(lidar_fps, 2),
        "lidar_bytes_per_sec": round(lidar_bps, 1),
        "server_frames": server_frames,
        "delivery_rate": round(delivery_rate, 4),
        "latency_mean_ms": round(mean_lat, 2),
        "latency_min_ms": round(min_lat, 2),
        "latency_max_ms": round(max_lat, 2),
        "inference_mean_ms": round(mean_infer, 2),
        "total_detections": total_det,
    }


if __name__ == "__main__":
    summary = main()
    if summary:
        out = Path(sys.argv[0]).parent / "measurement_summary.json"
        out.write_text(json.dumps(summary, indent=2))
