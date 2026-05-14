#!/usr/bin/env python3
"""
LIDAR streaming server (LD19 / LDLidar via /dev/ttyTHS1).

Reads serial packets from the LD19, assembles full 360° scans, and
broadcasts each scan as a length-prefixed JSON message over TCP.

LD19 packet layout (47 bytes total):
    0x54           header
    0x2C           VerLen (high 3 bits = ver, low 5 bits = num points = 12)
    [2]  speed       (deg/s, LE)
    [2]  start_angle (0.01°, LE)
    [12 * 3]   measurement points: [dist_lo, dist_hi, confidence]
    [2]  end_angle   (0.01°, LE)
    [2]  timestamp   (ms, LE)
    [1]  CRC

JSON scan payload:
    {
        "scan_id": int,
        "send_timestamp": float,
        "rotation_speed_deg_s": int,
        "num_points": int,
        "points": [[angle_deg, distance_mm, confidence], ...]
    }

Runs on the HOST (LIDAR is wired to host UART, not the container).

Usage:
    python3 ~/Desktop/lidar_stream.py
    python3 ~/Desktop/lidar_stream.py --port 8092 --log /tmp/lidar.csv
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

import serial

PORT = "/dev/ttyTHS1"
BAUD = 230_400

PKT_LEN = 47          # full packet incl. 0x54 header
POINTS_PER_PKT = 12

_shutdown = threading.Event()
_latest_scan_lock = threading.Lock()
_latest_scan_payload = None        # bytes (length-prefixed JSON ready to send)
_scan_event = threading.Event()


def read_lidar_loop(serial_port, log_writer, log_file):
    """Read LD19 packets, assemble 360° scans, publish to broadcast queue."""
    global _latest_scan_payload

    try:
        ser = serial.Serial(serial_port, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"[LIDAR] Cannot open {serial_port}: {e}", file=sys.stderr)
        _shutdown.set()
        return

    print(f"[LIDAR] Connected to {serial_port}")

    scan_points = []        # [[angle_deg, distance_mm, confidence], ...]
    last_end_angle = None
    scan_id = 0

    try:
        while not _shutdown.is_set():
            # Find packet header 0x54
            b = ser.read(1)
            if not b or b != b"\x54":
                continue
            b = ser.read(1)
            if not b or b != b"\x2C":
                continue

            payload = ser.read(45)
            if len(payload) != 45:
                continue

            speed = int.from_bytes(payload[0:2], "little")
            start_angle = int.from_bytes(payload[2:4], "little") / 100.0
            end_angle = int.from_bytes(payload[40:42], "little") / 100.0

            # 12 measurement points spread linearly between start and end
            if end_angle < start_angle:
                # wrap-around within packet
                arc = end_angle + 360.0 - start_angle
            else:
                arc = end_angle - start_angle
            step = arc / max(POINTS_PER_PKT - 1, 1)

            for i in range(POINTS_PER_PKT):
                p = payload[4 + i * 3:4 + i * 3 + 3]
                dist = int.from_bytes(p[0:2], "little")
                conf = p[2]
                ang = (start_angle + step * i) % 360.0
                scan_points.append([round(ang, 2), dist, conf])

            # Detect full-scan boundary: end_angle wraps below previous end_angle
            if last_end_angle is not None and end_angle < last_end_angle - 10.0:
                send_ts = time.time()
                scan_id += 1
                msg = {
                    "scan_id": scan_id,
                    "send_timestamp": send_ts,
                    "rotation_speed_deg_s": speed,
                    "num_points": len(scan_points),
                    "points": scan_points,
                }
                payload_bytes = json.dumps(msg).encode("utf-8")
                framed = struct.pack(">I", len(payload_bytes)) + payload_bytes

                with _latest_scan_lock:
                    _latest_scan_payload = framed
                _scan_event.set()
                _scan_event.clear()

                if log_writer:
                    log_writer.writerow([
                        f"{send_ts:.6f}", scan_id,
                        len(scan_points), len(payload_bytes),
                        speed,
                    ])
                    if scan_id % 10 == 0:
                        log_file.flush()

                if scan_id % 10 == 0:
                    print(f"  Scan {scan_id:>5d}  |  pts={len(scan_points):>4d}  |  "
                          f"bytes={len(payload_bytes):>5d}  |  "
                          f"speed={speed} deg/s    ", end="\r")

                scan_points = []

            last_end_angle = end_angle

    except serial.SerialException as e:
        print(f"\n[LIDAR] Serial error: {e}", file=sys.stderr)
    finally:
        ser.close()


def run_tcp_broadcast_server(port):
    """Accept TCP connections and broadcast each new scan to all clients."""
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
                print(f"\n[LIDAR] Client connected: {addr}")
            except socket.timeout:
                continue
            except OSError:
                break

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()

    while not _shutdown.is_set():
        _scan_event.wait(timeout=1.0)
        with _latest_scan_lock:
            payload = _latest_scan_payload
        if payload is None:
            continue

        dead = []
        with clients_lock:
            for c in clients:
                try:
                    c.sendall(payload)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(c)
            for c in dead:
                clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    srv.close()


def main():
    parser = argparse.ArgumentParser(description="LD19 LIDAR streaming server")
    parser.add_argument("--serial", default=PORT,
                        help=f"Serial device (default: {PORT})")
    parser.add_argument("--port", type=int, default=8092,
                        help="TCP broadcast port (default: 8092)")
    parser.add_argument("--log", default=None,
                        help="Per-scan CSV log path (optional)")
    args = parser.parse_args()

    log_file = None
    log_writer = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "send_timestamp", "scan_id", "num_points",
            "json_bytes", "rotation_speed_deg_s",
        ])
        print(f"LIDAR log: {log_path}")

    def handle_signal(sig, _frame):
        _shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Starting LIDAR broadcast on TCP :{args.port}")

    reader_thread = threading.Thread(
        target=read_lidar_loop,
        args=(args.serial, log_writer, log_file),
        daemon=True)
    reader_thread.start()

    try:
        run_tcp_broadcast_server(args.port)
    finally:
        _shutdown.set()
        reader_thread.join(timeout=3)
        if log_file:
            log_file.close()
        print("\n[LIDAR] Stopped.")


if __name__ == "__main__":
    main()
