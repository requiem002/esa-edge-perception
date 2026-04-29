#!/usr/bin/env python3
"""
Network degradation test runner for CV perception experiments.

Runs on the HOST (outside Docker). For each network profile, applies
simulated constraints via a proxy layer and measures perception quality.

The Tegra kernel (5.15.136-tegra) does not include sch_netem, so this
script implements network emulation in userspace. A proxy sits between
the streaming server and client, applying configurable:
  - Latency (delay added to each forwarded packet)
  - Bandwidth limiting (bytes/sec throttle)
  - Packet loss (random drop probability)

This approach is deterministic and per-frame measurable, which is
preferable for thesis experiments.

Usage:
    python3 ~/Desktop/network_test.py
    python3 ~/Desktop/network_test.py --duration 30 --profiles all
"""

import argparse
import csv
import json
import math
import os
import queue
import random
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── Network profiles ─────────────────────────────────────────────────
# Each profile simulates a network condition between the robot (server)
# and the remote operator (client).
#
# Fields:
#   delay_ms:      one-way latency added per packet (ms)
#   bandwidth_bps: maximum throughput in bytes per second
#   loss_pct:      probability of dropping a packet [0-100]

PROFILES = {
    "1_baseline": {
        "label": "Baseline (unconstrained)",
        "delay_ms": 0,
        "bandwidth_bps": 0,        # 0 = unlimited
        "loss_pct": 0.0,
    },
    "2_good_5g": {
        "label": "Good 5G (50 Mbps, 20ms)",
        "delay_ms": 20,
        "bandwidth_bps": 50_000_000 // 8,   # 50 Mbps
        "loss_pct": 0.0,
    },
    "3_congested_5g": {
        "label": "Congested 5G (10 Mbps, 50ms)",
        "delay_ms": 50,
        "bandwidth_bps": 10_000_000 // 8,   # 10 Mbps
        "loss_pct": 0.0,
    },
    "4_poor_5g": {
        "label": "Poor 5G (5 Mbps, 100ms, 1% loss)",
        "delay_ms": 100,
        "bandwidth_bps": 5_000_000 // 8,    # 5 Mbps
        "loss_pct": 1.0,
    },
    "5_satellite_ntn": {
        "label": "Satellite NTN (1 Mbps, 600ms, 2% loss)",
        "delay_ms": 600,
        "bandwidth_bps": 1_000_000 // 8,    # 1 Mbps
        "loss_pct": 2.0,
    },
    "6_degraded_ntn": {
        "label": "Degraded NTN (0.5 Mbps, 1200ms, 5% loss)",
        "delay_ms": 1200,
        "bandwidth_bps": 500_000 // 8,      # 0.5 Mbps
        "loss_pct": 5.0,
    },
}

# ── Constrained TCP proxy ────────────────────────────────────────────

class ConstrainedProxy:
    """TCP proxy that applies network constraints between source and dest.

    Applies to the metadata channel (port 8091). The MJPEG video stream
    passes through its own HTTP proxy with the same constraints.
    """

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
        self.bytes_forwarded = 0
        self.packets_dropped = 0
        self.packets_forwarded = 0

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

            # Connect to actual server
            try:
                upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                upstream.connect((self.dest_host, self.dest_port))
            except (ConnectionRefusedError, OSError):
                client_conn.close()
                continue

            # Forward in both directions with constraints
            t1 = threading.Thread(
                target=self._forward,
                args=(upstream, client_conn, True),
                daemon=True)
            t2 = threading.Thread(
                target=self._forward,
                args=(client_conn, upstream, False),
                daemon=True)
            t1.start()
            t2.start()

        self._server.close()

    def _forward(self, src, dst, apply_constraints):
        """Forward data from src to dst, optionally applying constraints."""
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
                    # Packet loss
                    if self.loss_pct > 0 and random.random() * 100 < self.loss_pct:
                        self.packets_dropped += 1
                        continue

                    # Latency
                    if self.delay_ms > 0:
                        time.sleep(self.delay_ms / 1000.0)

                    # Bandwidth limiting
                    if self.bandwidth_bps > 0:
                        transfer_time = len(data) / self.bandwidth_bps
                        time.sleep(transfer_time)

                    self.packets_forwarded += 1
                    self.bytes_forwarded += len(data)

                try:
                    dst.sendall(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        except Exception:
            pass
        finally:
            try:
                src.close()
            except OSError:
                pass
            try:
                dst.close()
            except OSError:
                pass


class ConstrainedHTTPProxy:
    """HTTP proxy for MJPEG stream with bandwidth/latency/loss constraints.

    Fetches MJPEG from upstream, applies constraints, and re-serves to client.
    """

    def __init__(self, listen_port, upstream_url,
                 delay_ms=0, bandwidth_bps=0, loss_pct=0.0):
        self.listen_port = listen_port
        self.upstream_url = upstream_url
        self.delay_ms = delay_ms
        self.bandwidth_bps = bandwidth_bps
        self.loss_pct = loss_pct
        self._stop = threading.Event()
        self._thread = None
        self.frames_forwarded = 0
        self.frames_dropped = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        """Read MJPEG from upstream, apply constraints, serve to local clients."""
        import cv2 as _cv2
        from http.server import HTTPServer, BaseHTTPRequestHandler

        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self_h):
                if self_h.path != "/stream":
                    self_h.send_error(404)
                    return

                cap = _cv2.VideoCapture(proxy.upstream_url)
                if not cap.isOpened():
                    self_h.send_error(502, "Cannot connect to upstream")
                    return

                self_h.send_response(200)
                self_h.send_header("Content-Type",
                                   "multipart/x-mixed-replace; boundary=frame")
                self_h.end_headers()

                try:
                    while not proxy._stop.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            time.sleep(0.05)
                            continue

                        # Packet loss
                        if (proxy.loss_pct > 0 and
                                random.random() * 100 < proxy.loss_pct):
                            proxy.frames_dropped += 1
                            continue

                        # Latency
                        if proxy.delay_ms > 0:
                            time.sleep(proxy.delay_ms / 1000.0)

                        ok, jpeg = _cv2.imencode(".jpg", frame,
                                                  [_cv2.IMWRITE_JPEG_QUALITY, 70])
                        if not ok:
                            continue
                        jpeg_bytes = jpeg.tobytes()

                        # Bandwidth limiting
                        if proxy.bandwidth_bps > 0:
                            transfer_time = len(jpeg_bytes) / proxy.bandwidth_bps
                            time.sleep(transfer_time)

                        try:
                            self_h.wfile.write(b"--frame\r\n")
                            self_h.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self_h.wfile.write(
                                f"Content-Length: {len(jpeg_bytes)}\r\n".encode())
                            self_h.wfile.write(b"\r\n")
                            self_h.wfile.write(jpeg_bytes)
                            self_h.wfile.write(b"\r\n")
                            proxy.frames_forwarded += 1
                        except (BrokenPipeError, ConnectionResetError):
                            break
                except Exception:
                    pass
                finally:
                    cap.release()

            def log_message(self_h, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", self.listen_port), Handler)
        server.timeout = 1.0
        while not self._stop.is_set():
            server.handle_request()
        server.server_close()


# ── Main experiment runner ───────────────────────────────────────────

def run_client(host, port, meta_port, duration, output_csv):
    """Run stream_client.py as a subprocess and return its exit code."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "stream_client.py"),
        "--host", host,
        "--port", str(port),
        "--meta-port", str(meta_port),
        "--duration", str(duration),
        "--output", str(output_csv),
    ]
    result = subprocess.run(cmd, timeout=duration + 30)
    return result.returncode


def parse_client_csv(csv_path):
    """Parse the client's measurement CSV and compute summary stats."""
    rows = []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        return {}

    if not rows:
        return {
            "meta_messages": 0, "meta_fps": 0, "delivery_rate": 0,
            "latency_mean_ms": 0, "latency_min_ms": 0, "latency_max_ms": 0,
            "latency_p95_ms": 0, "inference_mean_ms": 0,
            "total_detections": 0,
        }

    latencies = [float(r["latency_ms"]) for r in rows]
    infer_times = [float(r["inference_time_ms"]) for r in rows]
    det_counts = [int(r["num_detections"]) for r in rows]

    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95)

    # Compute received FPS from timestamps
    if len(rows) >= 2:
        t_first = float(rows[0]["recv_timestamp"])
        t_last = float(rows[-1]["recv_timestamp"])
        duration = t_last - t_first
        meta_fps = len(rows) / duration if duration > 0 else 0
    else:
        meta_fps = 0

    # Delivery rate: messages received / server frame number range
    frame_numbers = [int(r["frame_number"]) for r in rows]
    if frame_numbers:
        fn_min, fn_max = min(frame_numbers), max(frame_numbers)
        expected = fn_max - fn_min + 1
        delivery_rate = len(rows) / expected if expected > 0 else 0
    else:
        delivery_rate = 0

    return {
        "meta_messages": len(rows),
        "meta_fps": round(meta_fps, 2),
        "delivery_rate": round(delivery_rate, 4),
        "latency_mean_ms": round(sum(latencies) / len(latencies), 2),
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
        "latency_p95_ms": round(latencies_sorted[p95_idx], 2),
        "inference_mean_ms": round(sum(infer_times) / len(infer_times), 2),
        "total_detections": sum(det_counts),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Network degradation experiment runner")
    parser.add_argument("--duration", type=int, default=60,
                        help="Seconds per profile (default: 60)")
    parser.add_argument("--profiles", default="all",
                        help="Comma-separated profile keys or 'all'")
    parser.add_argument("--server-host", default="127.0.0.1",
                        help="stream_server.py host (default: 127.0.0.1)")
    parser.add_argument("--server-port", type=int, default=8090,
                        help="stream_server MJPEG port")
    parser.add_argument("--server-meta-port", type=int, default=8091,
                        help="stream_server metadata port")
    parser.add_argument("--output-dir", default="network_results",
                        help="Results directory")
    parser.add_argument("--settle-time", type=int, default=5,
                        help="Seconds to wait after applying constraints")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repetitions per profile (default: 1)")
    args = parser.parse_args()

    # Proxy ports (client connects here instead of directly to server)
    proxy_video_port = 9090
    proxy_meta_port = 9091

    # Resolve profiles
    if args.profiles == "all":
        profile_keys = list(PROFILES.keys())
    else:
        profile_keys = [k.strip() for k in args.profiles.split(",")]
        for k in profile_keys:
            if k not in PROFILES:
                sys.exit(f"Unknown profile: {k}. Available: {list(PROFILES.keys())}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 65}")
    print(f"  NETWORK DEGRADATION EXPERIMENT")
    print(f"  {len(profile_keys)} profiles, {args.duration}s each, "
          f"{args.repeats} repeat(s)")
    print(f"  Server: {args.server_host}:{args.server_port}")
    print(f"  NOTE: Constraints applied via userspace proxy")
    print(f"        (Tegra kernel lacks sch_netem)")
    print(f"{'=' * 65}\n")

    # Verify server is reachable
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect((args.server_host, args.server_port))
        sock.close()
        print("[OK] Server reachable\n")
    except (ConnectionRefusedError, socket.timeout, OSError):
        sys.exit(f"[ERROR] Cannot reach server at "
                 f"{args.server_host}:{args.server_port}\n"
                 f"Start it first:\n"
                 f"  docker exec -it yolo-saad python3 "
                 f"/workspace/stream_server.py")

    all_summaries = []

    # Metric keys used for mean/stddev aggregation
    METRIC_KEYS = [
        "meta_messages", "meta_fps", "delivery_rate",
        "latency_mean_ms", "latency_min_ms", "latency_max_ms",
        "latency_p95_ms", "inference_mean_ms", "total_detections",
    ]

    total_runs = len(profile_keys) * args.repeats
    run_num = 0

    for i, key in enumerate(profile_keys, 1):
        profile = PROFILES[key]
        profile_dir = out_dir / key
        profile_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─' * 65}")
        print(f"  [{i}/{len(profile_keys)}] {profile['label']}")
        print(f"  Delay: {profile['delay_ms']}ms  |  "
              f"BW: {profile['bandwidth_bps'] * 8 / 1_000_000:.1f} Mbps  |  "
              f"Loss: {profile['loss_pct']}%")
        if args.repeats > 1:
            print(f"  Repeats: {args.repeats}")
        print(f"{'─' * 65}")

        rep_summaries = []

        for rep in range(1, args.repeats + 1):
            run_num += 1

            if args.repeats > 1:
                csv_path = profile_dir / f"measurements_rep{rep}.csv"
                print(f"\n  --- Repeat {rep}/{args.repeats} "
                      f"(run {run_num}/{total_runs}) ---")
            else:
                csv_path = profile_dir / "measurements.csv"

            # Start constrained proxies
            video_proxy = ConstrainedHTTPProxy(
                listen_port=proxy_video_port,
                upstream_url=f"http://{args.server_host}:{args.server_port}/stream",
                delay_ms=profile["delay_ms"],
                bandwidth_bps=profile["bandwidth_bps"],
                loss_pct=profile["loss_pct"],
            )
            meta_proxy = ConstrainedProxy(
                listen_port=proxy_meta_port,
                dest_host=args.server_host,
                dest_port=args.server_meta_port,
                delay_ms=profile["delay_ms"],
                bandwidth_bps=profile["bandwidth_bps"],
                loss_pct=profile["loss_pct"],
            )

            video_proxy.start()
            meta_proxy.start()

            # Settle
            print(f"  Settling for {args.settle_time}s...")
            time.sleep(args.settle_time)

            # Run client
            print(f"  Running client for {args.duration}s...")
            rc = run_client(
                host="127.0.0.1",
                port=proxy_video_port,
                meta_port=proxy_meta_port,
                duration=args.duration,
                output_csv=str(csv_path),
            )

            # Stop proxies
            video_proxy.stop()
            meta_proxy.stop()

            if rc != 0:
                print(f"  [WARN] Client exited with code {rc}")

            rep_summary = parse_client_csv(csv_path)
            rep_summaries.append(rep_summary)

            print(f"  Results: FPS={rep_summary.get('meta_fps', 0):.1f}  "
                  f"Latency={rep_summary.get('latency_mean_ms', 0):.0f}ms  "
                  f"Delivery={rep_summary.get('delivery_rate', 0):.1%}")

            # Brief cooldown between repeats
            if rep < args.repeats:
                print(f"  Cooldown 3s...")
                time.sleep(3)

        # Aggregate across repeats: mean and stddev
        summary = {}
        for mk in METRIC_KEYS:
            vals = [r.get(mk, 0) for r in rep_summaries]
            summary[mk] = round(statistics.mean(vals), 2)
            if args.repeats > 1:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                summary[f"{mk}_std"] = round(sd, 2)

        summary["profile"] = key
        summary["label"] = profile["label"]
        summary["delay_ms"] = profile["delay_ms"]
        summary["bandwidth_mbps"] = round(
            profile["bandwidth_bps"] * 8 / 1_000_000, 1)
        summary["loss_pct"] = profile["loss_pct"]

        # Save profile config with all repeat data
        with open(profile_dir / "profile.json", "w") as f:
            json.dump({
                "profile": profile,
                "summary": summary,
                "repeats": rep_summaries,
            }, f, indent=2)

        all_summaries.append(summary)

    # Write summary CSV
    summary_csv = out_dir / "summary.csv"
    if all_summaries:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            for s in all_summaries:
                writer.writerow(s)

    print(f"\n\n{'=' * 65}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Summary: {summary_csv}")
    if args.repeats > 1:
        print(f"  Per-profile data: {out_dir}/*/measurements_repN.csv")
    else:
        print(f"  Per-profile data: {out_dir}/*/measurements.csv")
    print()

    # Print results table
    if args.repeats > 1:
        print(f"  {'Profile':<30s} {'FPS':>10s} {'Lat(ms)':>14s} "
              f"{'Delivery':>12s} {'Loss':>6s}")
        print(f"  {'─' * 30} {'─' * 10} {'─' * 14} {'─' * 12} {'─' * 6}")
        for s in all_summaries:
            fps_str = f"{s.get('meta_fps', 0):.1f}±{s.get('meta_fps_std', 0):.1f}"
            lat_str = f"{s.get('latency_mean_ms', 0):.0f}±{s.get('latency_mean_ms_std', 0):.0f}"
            dr_str = f"{s.get('delivery_rate', 0):.1%}±{s.get('delivery_rate_std', 0):.1%}"
            print(f"  {s['label']:<30s} {fps_str:>10s} {lat_str:>14s} "
                  f"{dr_str:>12s} {s['loss_pct']:>5.1f}%")
    else:
        print(f"  {'Profile':<30s} {'FPS':>6s} {'Lat(ms)':>8s} "
              f"{'Delivery':>9s} {'Loss':>6s}")
        print(f"  {'─' * 30} {'─' * 6} {'─' * 8} {'─' * 9} {'─' * 6}")
        for s in all_summaries:
            print(f"  {s['label']:<30s} "
                  f"{s.get('meta_fps', 0):>6.1f} "
                  f"{s.get('latency_mean_ms', 0):>8.1f} "
                  f"{s.get('delivery_rate', 0):>8.1%} "
                  f"{s['loss_pct']:>5.1f}%")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
