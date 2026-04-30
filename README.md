# Edge-AI Perception for Robotic Operations over 5G/NTN

Real-time object detection on a Unitree Go2 quadruped robot, running entirely at the edge on an NVIDIA Jetson AGX Orin. Designed for teleoperation over degraded 5G and satellite (NTN) links — the robot sees and decides locally; only structured results are sent to the operator.

**Platform:** Unitree Go2 + Jetson AGX Orin (JetPack R36, CUDA 12.2)
**Model:** YOLOv11-nano via TensorRT (~25 FPS at 640x480)
**Streams:** MJPEG video (8090), JSON detections (8091), LiDAR scans (8092)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        JETSON AGX ORIN (Edge)                          │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                 Docker: yolo-saad (--runtime nvidia)             │  │
│  │                                                                  │  │
│  │   /dev/video0 ──► inference.py ──► detect_*.avi + detections.csv │  │
│  │        │              │                                          │  │
│  │        │              ▼                                          │  │
│  │        └────► stream_server.py                                   │  │
│  │                  │         │                                      │  │
│  │            :8090 MJPEG  :8091 JSON                               │  │
│  └──────────────│─────────────│─────────────────────────────────────┘  │
│                 │             │                                         │
│   /dev/ttyTHS1 ──► lidar_stream.py ──► :8092 JSON (360° scans)        │
│                 │             │              │                          │
└─────────────────│─────────────│──────────────│─────────────────────────┘
                  │             │              │
            ┌─────▼─────────────▼──────────────▼─────┐
            │         5G / NTN / Satellite            │
            │  (emulated by network_test.py proxy)    │
            └─────┬─────────────┬──────────────┬─────┘
                  │             │              │
┌─────────────────▼─────────────▼──────────────▼─────────────────────────┐
│                      OPERATOR STATION (Remote)                         │
│                                                                        │
│   stream_client.py ──► Video FPS, metadata rate, LiDAR Hz, latency    │
│   Browser           ──► http://<jetson-ip>:8090/stream (live view)     │
└────────────────────────────────────────────────────────────────────────┘
```

### Three-tier stream design

The key insight: under bandwidth pressure, sacrifice pixels first, keep structured data last.

```
                    Bandwidth available ──►
                    Full        Degraded      Satellite NTN
  Video  (8090)     28 FPS  ──►  5 FPS   ──►  0.6 FPS     (~330 KiB/s → collapses)
  LiDAR  (8092)     10 Hz   ──►  6 Hz    ──►  1.8 Hz      (~90 KiB/s → degrades)
  Meta   (8091)     27 msg/s ──► 27 msg/s ──► 27 msg/s    (~5 KiB/s → survives)
```

Detection metadata (class, confidence, bounding box, timestamp) is ~200 bytes per frame. Even on a 0.5 Mbps degraded satellite link, the operator maintains full situational awareness through structured data while video becomes a periodic snapshot.

### Data flow

```
Camera frame
    │
    ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│  TensorRT    │────►│  Post-filter  │────►│  CSV logger      │
│  inference   │     │  (8 classes)  │     │  (detections.csv)│
│  (~25ms)     │     └───────┬───────┘     └──────────────────┘
└──────────────┘             │
                             ▼
                    ┌─────────────────┐
                    │  Annotated      │──── detect_*.avi (local)
                    │  frame          │──── :8090 MJPEG  (remote)
                    └─────────────────┘──── :8091 JSON   (remote)
```

---

## Quick start

```bash
# Live inference (with display window)
~/Desktop/run.sh

# Headless inference (saves video + CSV, default for boot service)
~/Desktop/run.sh --no-display

# Start streaming server (for remote operator or network experiments)
~/Desktop/run.sh --stream

# Full network degradation experiment (6 profiles x 60s x 3 repeats)
~/Desktop/run.sh --experiment

# Status dashboard
~/Desktop/run.sh --status

# Stop everything
~/Desktop/stop.sh
```

### Useful flags

```bash
# Custom confidence threshold
~/Desktop/run.sh --conf 0.4

# Use a video file instead of live camera
~/Desktop/run.sh --source /workspace/test.mp4

# Lower thermal throttle threshold
~/Desktop/run.sh --no-display --thermal-throttle 80

# Watch the systemd service logs
journalctl -u yolo-inference.service -f
```

---

## File map

| File | Location | Purpose |
|------|----------|---------|
| `inference.py` | Container | Core YOLO detection loop with thermal management, disk guards, output rotation |
| `stream_server.py` | Container | MJPEG + TCP metadata server for remote viewing |
| `stream_client.py` | Host | Measures video FPS, metadata rate, latency from operator side |
| `lidar_stream.py` | Host | LD19 LiDAR serial reader, broadcasts 360-degree scans on TCP :8092 |
| `network_test.py` | Host | Orchestrates network degradation experiments (6 profiles, userspace proxy) |
| `analyse_results.py` | Host | Generates 6 thesis-quality plots from experiment data |
| `status.py` | Host | Quick dashboard: container, GPU temp, disk, process status |
| `run.sh` | Host | Universal launcher (inference / stream / experiment / status) |
| `stop.sh` | Host | Kills all pipeline processes cleanly |
| `yolo-inference.service` | systemd | Auto-starts headless inference on boot |
| `lidar_test.py` | Host | Low-level LD19 packet decode reference |

---

## Network experiment

The Tegra kernel (5.15.136-tegra) lacks `sch_netem`, so network emulation is implemented as a **userspace proxy** that applies delay, bandwidth limiting, and packet loss per-frame. This is deterministic and individually measurable — preferable for thesis experiments.

### Profiles

| # | Profile | Delay | Bandwidth | Loss |
|---|---------|-------|-----------|------|
| 1 | Baseline | 0 ms | unlimited | 0% |
| 2 | Good 5G | 10 ms | 50 Mbps | 0% |
| 3 | Congested 5G | 50 ms | 10 Mbps | 1% |
| 4 | Poor 5G | 100 ms | 2 Mbps | 2% |
| 5 | Satellite NTN | 600 ms | 2 Mbps | 1% |
| 6 | Degraded NTN | 1200 ms | 0.5 Mbps | 5% |

### Running

```bash
# Full experiment (recommended)
~/Desktop/run.sh --experiment

# Quick 2-profile test
python3 ~/Desktop/network_test.py --duration 10 --profiles 1_baseline,6_degraded_ntn

# Generate plots from existing data
python3 ~/Desktop/analyse_results.py --input network_results/full_experiment_v2/summary.csv
```

Results land in `~/Desktop/network_results/` (gitignored).

---

## Deployment

### Docker container

```bash
docker run -d --name yolo-saad --ipc=host --runtime nvidia --network host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --device /dev/video0 --device /dev/ttyTHS1 \
    -v ~/Desktop:/workspace \
    yolo-orin:v2 sleep infinity
```

### TensorRT engine

Engines are device-specific — always rebuild on the target Jetson:

```bash
docker exec yolo-saad yolo export model=/workspace/yolo26n.pt format=engine
```

### Auto-start on boot

The `yolo-inference.service` is already installed and enabled. It starts the container and runs headless inference automatically. Verify with:

```bash
sudo systemctl reboot
# after reboot:
journalctl -u yolo-inference.service -f --since "5 min ago"
```

### LiDAR

The LD19 connects via `/dev/ttyTHS1` at 230,400 baud on the **host** (not inside Docker). Start manually or wrap in its own systemd service:

```bash
python3 ~/Desktop/lidar_stream.py              # foreground
python3 ~/Desktop/lidar_stream.py --log scan.csv  # with logging
```

---

## Thermal management

| Threshold | Action |
|-----------|--------|
| GPU < 70 C | Green overlay, normal operation |
| GPU 70-85 C | Yellow overlay, normal operation |
| GPU > 85 C | Red overlay, skip every other frame (throttle) |
| GPU > 95 C | Hard halt — inference stops |

Configurable via `--thermal-throttle` and `--thermal-halt`.

---

## Target classes

Post-inference filter (no retraining needed to change):

`person`, `dog`, `cat`, `car`, `bicycle`, `motorcycle`, `truck`, `bus`

Modify `RELEVANT_CLASSES` in `inference.py` to add/remove classes.

---

## Requirements

See [`requirements.md`](requirements.md) for 56 formal requirements across 10 categories (44 Done, 5 Partial, 7 TODO). See [`deployment_checklist.md`](deployment_checklist.md) for Go2 physical deployment steps.

---

## Repository

Only source code and documentation are tracked. Models (`.pt`, `.engine`), media (`.avi`, `.mp4`), data (`.csv`), and experiment results are gitignored.

```
~/Desktop/
  ├── *.py, *.sh, *.md       ← tracked
  ├── yolo-inference.service  ← tracked
  ├── .gitignore              ← tracked
  ├── output/                 ← gitignored (video + CSV per session)
  ├── network_results/        ← gitignored (experiment data + plots)
  ├── runs/                   ← gitignored (YOLO CLI output)
  └── *.pt, *.engine, *.onnx  ← gitignored (models)
```
