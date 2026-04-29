#!/usr/bin/env python3
"""
YOLO detection for autonomous Jetson deployment (robot dog).

Features:
- Filters to outdoor-relevant classes only
- Auto-recovers on camera disconnect with exponential backoff
- GPU temperature monitoring and overlay
- Output file rotation (keeps last 10 sessions, deletes oldest)
- Disk space guard (stops recording if < 200MB free)

Usage (inside container):
    python3 /workspace/inference.py
    python3 /workspace/inference.py --model /workspace/yolo26n.engine --source 0
    python3 /workspace/inference.py --conf 0.4 --output-dir /workspace/output
"""

import argparse
import csv
import os
import shutil
import signal
import stat
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# Classes relevant to outdoor / robot-dog deployment
RELEVANT_CLASSES = {"person", "dog", "cat", "car", "bicycle", "motorcycle", "truck", "bus"}

# BGR colours for each class on the annotated frame
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

# Disk space threshold (bytes) — stop recording below this
DISK_MIN_BYTES = 200 * 1024 * 1024  # 200 MB

# Max output sessions to keep on disk
MAX_OUTPUT_FILES = 10


def resolve_model(model_path):
    """Try the requested model; fall back to .pt if a .engine file is missing."""
    p = Path(model_path)
    if p.exists():
        return str(p)
    if p.suffix == ".engine":
        fallback = p.with_suffix(".pt")
        if fallback.exists():
            print(f"[WARN] Engine not found: {p}")
            print(f"       Falling back to: {fallback}")
            return str(fallback)
    print(f"[ERROR] Model not found: {model_path}")
    sys.exit(1)


def read_gpu_temp():
    """Read GPU temperature (Celsius) from sysfs thermal zones."""
    thermal_base = Path("/sys/devices/virtual/thermal/")
    for zone in sorted(thermal_base.glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == "gpu-thermal":
                return int((zone / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError, TypeError):
            continue
    return None


def open_camera(source):
    """Open camera, retrying with exponential backoff until success or signal."""
    attempt = 0
    while True:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        cap.release()
        attempt += 1
        wait = min(2 ** attempt, 30)
        print(f"[WARN] Camera open failed (attempt {attempt}), retrying in {wait}s...")
        time.sleep(wait)


def make_world_writable(path):
    """Set a file/dir to world-writable so the host user can manage it."""
    try:
        p = Path(path)
        if p.exists():
            p.chmod(p.stat().st_mode | stat.S_IWOTH | stat.S_IROTH)
    except OSError:
        pass


def rotate_output_files(output_dir, keep=MAX_OUTPUT_FILES):
    """Delete oldest output sessions beyond the keep limit."""
    videos = sorted(Path(output_dir).glob("detect_*.avi"),
                    key=lambda p: p.stat().st_mtime)
    while len(videos) > keep:
        oldest = videos.pop(0)
        csv_name = oldest.name.replace("detect_", "detections_").replace(".avi", ".csv")
        csv_pair = oldest.parent / csv_name
        oldest.unlink(missing_ok=True)
        csv_pair.unlink(missing_ok=True)
        print(f"[ROTATE] Deleted old output: {oldest.name}")


def main():
    parser = argparse.ArgumentParser(description="YOLO autonomous detection (robot dog)")
    parser.add_argument("--model", default="/workspace/yolo26n.engine",
                        help="Path to YOLO model (.engine or .pt)")
    parser.add_argument("--source", default="0",
                        help="Video source: device index (0) or file path")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", default="/workspace/output",
                        help="Directory for output video and CSV log")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable cv2.imshow (headless mode)")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save annotated video (default: True)")
    parser.add_argument("--thermal-throttle", type=float, default=85.0,
                        help="GPU temp (C) to start skipping frames (default: 85)")
    parser.add_argument("--thermal-halt", type=float, default=95.0,
                        help="GPU temp (C) to halt inference (default: 95)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    # Output directory and rotation
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_world_writable(out_dir)
    rotate_output_files(out_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = out_dir / f"detect_{timestamp_str}.avi"
    csv_path = out_dir / f"detections_{timestamp_str}.csv"

    # Load model
    model_path = resolve_model(args.model)
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Log which relevant classes this model supports
    matched = [c for c in RELEVANT_CLASSES if c in model.names.values()]
    if matched:
        print(f"Post-inference class filter: {sorted(matched)}")
    else:
        print("[WARN] No relevant classes found in model, showing all detections")

    # Open camera (retries until success)
    print(f"Opening camera source: {source}")
    cap = open_camera(source)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Source opened: {w}x{h} @ {cam_fps:.0f} fps")

    # Video writer
    writer = None
    recording = args.save_video
    if recording:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(video_path), fourcc, cam_fps, (w, h))

    # CSV logger
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "frame_number", "class", "confidence",
                         "x1", "y1", "x2", "y2"])

    # State
    class_counts = defaultdict(int)
    frame_count = 0
    fps_smooth = 0.0
    prev_time = time.time()
    camera_reconnects = 0
    thermal_throttle_frames = 0
    running = True

    def handle_signal(sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    gpu_temp = read_gpu_temp()
    temp_str = f", GPU {gpu_temp:.0f}C" if gpu_temp else ""
    print(f"\nInference started — conf={args.conf}, display={'off' if args.no_display else 'on'}{temp_str}")
    print(f"  Video  -> {video_path}")
    print(f"  CSV    -> {csv_path}")
    print("Press Ctrl+C or 'q' to stop.\n")

    while running:
        ret, frame = cap.read()
        if not ret:
            camera_reconnects += 1
            print(f"\n[WARN] Camera read failed at frame {frame_count}, reconnecting "
                  f"(recovery #{camera_reconnects})...")
            cap.release()
            cap = open_camera(source)
            if not running:
                break
            print(f"[INFO] Camera reconnected after recovery #{camera_reconnects}")
            continue

        frame_count += 1
        now = time.time()
        dt = now - prev_time
        prev_time = now

        instant_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

        # Thermal check (every 150 frames ~5s, or every frame when throttling)
        if frame_count % 150 == 0 or frame_count == 1 or thermal_throttle_frames > 0:
            gpu_temp = read_gpu_temp()

        # THM-4: halt at critical temperature
        if gpu_temp is not None and gpu_temp >= args.thermal_halt:
            print(f"\n[CRITICAL] GPU temperature {gpu_temp:.0f}C >= {args.thermal_halt:.0f}C halt threshold!")
            print("[CRITICAL] Stopping inference for thermal safety.")
            break

        # THM-3: throttle — skip every other frame when above threshold
        if gpu_temp is not None and gpu_temp >= args.thermal_throttle:
            thermal_throttle_frames += 1
            if thermal_throttle_frames % 2 == 0:
                continue  # skip this frame
            if thermal_throttle_frames == 1:
                print(f"\n[THERMAL] Throttling: GPU {gpu_temp:.0f}C >= "
                      f"{args.thermal_throttle:.0f}C, skipping every other frame")
        elif thermal_throttle_frames > 0:
            print(f"\n[THERMAL] Temperature {gpu_temp:.0f}C back below "
                  f"{args.thermal_throttle:.0f}C, resuming full rate")
            thermal_throttle_frames = 0

        # Run inference (no classes= arg — filtering done post-inference to
        # avoid TensorRT engine reloads that waste ~243MB GPU memory)
        results = model(frame, conf=args.conf, verbose=False)

        # Draw only RELEVANT_CLASSES detections on the frame
        annotated = frame.copy()
        ts = datetime.now().isoformat(timespec="milliseconds")
        det_count = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            if cls_name not in RELEVANT_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            # Draw box and label
            color = BOX_COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (ix1, iy1 - th - 6), (ix1 + tw, iy1), color, -1)
            cv2.putText(annotated, label, (ix1, iy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            det_count += 1
            class_counts[cls_name] += 1
            csv_writer.writerow([ts, frame_count, cls_name, f"{conf:.3f}",
                                 f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])

        # Overlay: FPS
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Overlay: detection count (filtered only)
        cv2.putText(annotated, f"Detections: {det_count}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Overlay: GPU temp (already read earlier in thermal check)
        if gpu_temp is not None:
            throt = args.thermal_throttle
            halt = args.thermal_halt
            color = (0, 0, 255) if gpu_temp >= throt else (0, 255, 255) if gpu_temp > 70 else (0, 255, 0)
            label = f"GPU: {gpu_temp:.0f}C"
            if thermal_throttle_frames > 0:
                label += " [THROTTLE]"
            cv2.putText(annotated, label, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Write video frame
        if writer:
            writer.write(annotated)

        # Live display
        if not args.no_display:
            cv2.imshow("YOLO Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Periodic: terminal status + disk check (every 30 frames)
        if frame_count % 30 == 0:
            total_det = sum(class_counts.values())
            temp_tag = f"  GPU: {gpu_temp:.0f}C" if gpu_temp else ""
            print(f"  Frame {frame_count:>6d}  |  FPS: {fps_smooth:>5.1f}  |  "
                  f"Det: {total_det}{temp_tag}    ", end="\r")

        # Disk space guard (every 300 frames ~10s)
        if frame_count % 300 == 0 and writer:
            disk_free = shutil.disk_usage(str(out_dir)).free
            if disk_free < DISK_MIN_BYTES:
                print(f"\n[WARN] Disk critically low ({disk_free // (1024*1024)}MB free), "
                      "stopping video recording")
                writer.release()
                writer = None
                csv_file.flush()

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    csv_file.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    # Make output files world-writable so host user can manage them
    make_world_writable(video_path)
    make_world_writable(csv_path)

    # Summary
    print(f"\n\n{'=' * 55}")
    print(f"  SESSION SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Frames processed : {frame_count}")
    print(f"  Average FPS      : {fps_smooth:.1f}")
    print(f"  Camera recoveries: {camera_reconnects}")
    if thermal_throttle_frames > 0:
        print(f"  Thermal throttled: {thermal_throttle_frames} frames skipped")
    print(f"  Model            : {model_path}")
    print(f"  Video saved      : {video_path}")
    print(f"  CSV log          : {csv_path}")
    gpu_temp = read_gpu_temp()
    if gpu_temp is not None:
        print(f"  GPU temperature  : {gpu_temp:.1f} C")
    print(f"\n  Detections by class:")
    print(f"  {'-' * 40}")
    if class_counts:
        for cls_name, count in sorted(class_counts.items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"    {cls_name:<20s} {count:>6d}")
    else:
        print("    (no detections)")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
