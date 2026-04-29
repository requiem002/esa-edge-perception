#!/usr/bin/env python3
"""
Streaming client — measures perception quality received from stream_server.py.

Runs on the HOST (outside Docker). Connects to the MJPEG video stream and
TCP metadata socket, measures latency, FPS, and detection delivery rate.

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

import cv2


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

            # Parse length-prefixed JSON messages
            while len(buf) >= 4:
                msg_len = struct.unpack(">I", buf[:4])[0]
                if len(buf) < 4 + msg_len:
                    break
                payload = buf[4:4 + msg_len].decode("utf-8")
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


def receive_video(url, frame_counts, stop_event):
    """Connect to MJPEG stream and count received frames."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[VIDEO] Could not open {url}")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_counts.append(time.time())

    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Stream client — measure perception under network constraints")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8090,
                        help="MJPEG port (default: 8090)")
    parser.add_argument("--meta-port", type=int, default=8091,
                        help="Metadata TCP port (default: 8091)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Measurement duration in seconds (default: 60)")
    parser.add_argument("--output", default="measurements.csv",
                        help="Output CSV path")
    parser.add_argument("--display", action="store_true",
                        help="Show received video (default: off)")
    args = parser.parse_args()

    stream_url = f"http://{args.host}:{args.port}/stream"

    meta_results = []
    video_timestamps = []
    stop_event = threading.Event()

    # Start receivers
    meta_thread = threading.Thread(
        target=receive_metadata,
        args=(args.host, args.meta_port, meta_results, stop_event),
        daemon=True)
    meta_thread.start()

    video_thread = threading.Thread(
        target=receive_video,
        args=(stream_url, video_timestamps, stop_event),
        daemon=True)
    video_thread.start()

    def handle_signal(sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Collecting for {args.duration}s from {args.host}...")
    print(f"  Video: {stream_url}")
    print(f"  Meta:  {args.host}:{args.meta_port}")
    print()

    t_start = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - t_start
        if elapsed >= args.duration:
            break

        # Progress
        n_meta = len(meta_results)
        n_video = len(video_timestamps)

        # Compute live FPS from video timestamps
        recent_ts = [t for t in video_timestamps if t > time.time() - 2.0]
        live_fps = len(recent_ts) / 2.0 if len(recent_ts) > 1 else 0.0

        print(f"  [{elapsed:5.0f}s / {args.duration}s]  "
              f"Video frames: {n_video}  |  Meta msgs: {n_meta}  |  "
              f"FPS: {live_fps:.1f}     ", end="\r")
        time.sleep(1.0)

    stop_event.set()
    meta_thread.join(timeout=3)
    video_thread.join(timeout=3)

    t_end = time.time()
    actual_duration = t_end - t_start

    # Write per-message CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Summary stats
    n_meta = len(meta_results)
    n_video = len(video_timestamps)

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
        frames_with_det = sum(1 for m in meta_results
                              if len(m.get("detections", [])) > 0)
    else:
        mean_lat = max_lat = min_lat = mean_infer = 0
        total_det = frames_with_det = 0

    video_fps = n_video / actual_duration if actual_duration > 0 else 0
    meta_fps = n_meta / actual_duration if actual_duration > 0 else 0

    # Compute frame delivery rate (meta frames / server frames)
    if meta_results:
        server_frames = meta_results[-1].get("frame_number", 0)
        delivery_rate = n_meta / server_frames if server_frames > 0 else 0
    else:
        server_frames = 0
        delivery_rate = 0

    print(f"\n\n{'=' * 60}")
    print(f"  MEASUREMENT SUMMARY  ({actual_duration:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  Video frames received  : {n_video}")
    print(f"  Video FPS (received)   : {video_fps:.1f}")
    print(f"  Metadata messages      : {n_meta}")
    print(f"  Metadata FPS           : {meta_fps:.1f}")
    print(f"  Server frames produced : {server_frames}")
    print(f"  Frame delivery rate    : {delivery_rate:.1%}")
    print(f"  Latency (mean/min/max) : {mean_lat:.1f} / {min_lat:.1f} / {max_lat:.1f} ms")
    print(f"  Mean inference time    : {mean_infer:.1f} ms")
    print(f"  Total detections       : {total_det}")
    print(f"  Frames with detections : {frames_with_det}")
    print(f"  CSV saved to           : {out_path}")
    print(f"{'=' * 60}\n")

    # Return summary dict (used by network_test.py)
    return {
        "duration_s": round(actual_duration, 1),
        "video_frames": n_video,
        "video_fps": round(video_fps, 2),
        "meta_messages": n_meta,
        "meta_fps": round(meta_fps, 2),
        "server_frames": server_frames,
        "delivery_rate": round(delivery_rate, 4),
        "latency_mean_ms": round(mean_lat, 2),
        "latency_min_ms": round(min_lat, 2),
        "latency_max_ms": round(max_lat, 2),
        "inference_mean_ms": round(mean_infer, 2),
        "total_detections": total_det,
        "frames_with_detections": frames_with_det,
    }


if __name__ == "__main__":
    summary = main()
    if summary:
        # Also save summary as JSON for programmatic use
        out = Path(sys.argv[0]).parent / "measurement_summary.json"
        import json as _json
        out.write_text(_json.dumps(summary, indent=2))
