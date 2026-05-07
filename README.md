# Edge-AI Perception for Robotic Operations over 5G/NTN

Real-time object detection on a Unitree Go2 quadruped robot, running entirely at the edge on an NVIDIA Jetson AGX Orin. Designed for teleoperation over degraded 5G and satellite (NTN) links — the robot sees and decides locally; only structured results are sent to the operator.

**Platform:** Unitree Go2 + Jetson AGX Orin (JetPack R36, CUDA 12.2)
**Model:** YOLOv11-nano via TensorRT (~25 FPS at 640×480)
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
│   dashboard_server.py ─────────────────────► :8080 Mission Control     │
│   sim_capture.py ──────────────────────────► :8093 MuJoCo MJPEG        │
│   go2_api.py ──────────────────────────────► :8094 Robot Control REST  │
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
│   Browser ──► http://<jetson-ip>:8080  (FENRIR-1 Mission Control)      │
│             ──► Live video, LiDAR polar, network analytics, robot ctrl │
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

---

## Quick start

```bash
# Live inference (with display window)
~/Desktop/run.sh

# Headless inference (saves video + CSV, default for boot service)
~/Desktop/run.sh --no-display

# Start streaming server (for remote operator or network experiments)
~/Desktop/run.sh --stream

# Full network degradation experiment (6 profiles × 60s × 3 repeats)
~/Desktop/run.sh --experiment

# Status dashboard (terminal)
~/Desktop/run.sh --status

# Mission Control web dashboard on :8080
~/Desktop/run.sh --dashboard

# MuJoCo simulator integration (Go2 control + sim view)
~/Desktop/run.sh --sim

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

## Mission Control Dashboard

A browser-based real-time dashboard served on port **:8080**. Access from any device on the same network (or via Tailscale): `http://<jetson-ip>:8080`

```bash
~/Desktop/run.sh --dashboard
```

### Panels

| Panel | Description |
|-------|-------------|
| **A — Camera Feed** | Live MJPEG stream from `stream_server.py`, with detection count and FPS overlay |
| **B — LiDAR Polar** | Real-time 360° point cloud rendered as polar canvas, last 3 scans with fade |
| **C — Controls** | Camera/LiDAR/logging toggles, network simulation, robot control |
| **D — Detections** | Per-frame class list with confidence bars, session totals |
| **F — Network Analytics** | Bandwidth bar, 3-tier degradation chart, E2E latency gauge, quality score ring, inferred link profile, frame loss indicator |
| **G — Sim View** | MuJoCo viewer capture (toggle via `[▶ SHOW SIM]`, requires `sim_capture.py`) |

### Offline / degraded state

- **All streams dead:** full-width amber blinking banner — `⚠ ALL STREAMS OFFLINE`; all panels dim to 30% opacity
- **Partial streams:** amber bar in status strip showing `DEGRADED — N/3 STREAMS ACTIVE`

### Network simulation

The controls panel includes six network profile buttons. Clicking one starts a `ConstrainedProxy` in the backend applying real delay/bandwidth/loss constraints to the metadata and LiDAR streams. The measured effect appears live in Panel F.

| Profile | Delay | Bandwidth | Loss |
|---------|-------|-----------|------|
| Baseline | 0 ms | unlimited | 0% |
| Good 5G | 20 ms | 50 Mbps | 0% |
| Congested 5G | 50 ms | 10 Mbps | 0% |
| Poor 5G | 100 ms | 5 Mbps | 1% |
| Satellite NTN | 600 ms | 1 Mbps | 2% |
| Degraded NTN | 1200 ms | 0.5 Mbps | 5% |

### Robot control

Stand and Crouch buttons send commands to `go2_api.py` (:8094) which executes `go2_cmd.py` as a subprocess. Target toggle switches between simulator (DOMAIN_ID=1) and real robot (DOMAIN_ID=0).

### Mock mode (no hardware)

```bash
python3 ~/Desktop/dashboard_server.py --mock
```

Generates synthetic telemetry oscillating through all six network profiles — use to develop/demo without Jetson hardware.

---

## MuJoCo Simulator Integration

The Unitree Go2 MuJoCo simulator (`~/unitree_mujoco/`) lets you test the full pipeline — including robot commands and dashboard — without the physical robot.

### Quick start

```bash
# Terminal 1: start the physics simulator (opens MuJoCo viewer window)
cd ~/unitree_mujoco/simulate_python
python3 unitree_mujoco.py

# Terminal 2: start dashboard services (sim view + robot API)
~/Desktop/run.sh --sim
# → Prompts you to confirm the viewer is open, then starts:
#   sim_capture.py  on :8093  (captures viewer via ffmpeg x11grab)
#   go2_api.py      on :8094  (sends DDS motor commands via go2_cmd.py)
```

Then open the dashboard and click **[▶ SHOW SIM]** to see the live 3D view.
Use **[▲ STAND]** / **[▼ CROUCH]** in the Robot Control panel to command the simulated robot.

### How it works

```
MuJoCo viewer (X11)
       │
sim_capture.py ──► ffmpeg x11grab ──► MJPEG on :8093
                                              │
                                     Dashboard Panel G

Dashboard Robot Control
       │
go2_api.py (:8094) ──► go2_cmd.py ──► DDS rt/lowcmd ──► MuJoCo bridge
                                                              │
                                                    unitree_sdk2py_bridge
                                                    (PD motor commands)
```

- **DOMAIN_ID = 1, INTERFACE = "lo"** for simulator (both `config.py` and `go2_cmd.py`)
- `go2_cmd.py` uses smooth S-curve interpolation over 1.5s for each pose transition
- `sim_capture.py` uses `xdotool` to find the window and `ffmpeg -f x11grab` to capture it

### Optional: YOLO on simulated camera

```bash
# Set up v4l2loopback virtual webcam (one-time)
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label=MuJoCo_Sim exclusive_caps=1

# Push MuJoCo front_cam frames to /dev/video2
python3 ~/unitree_mujoco/sim_camera_feed.py

# Inside container: run YOLO on the virtual camera
docker exec -it yolo-saad python3 /workspace/inference.py --source 2
```

---

## Standalone service scripts

| Script | Port | Purpose |
|--------|------|---------|
| `dashboard_server.py` | 8080 | Mission Control web dashboard |
| `sim_capture.py` | 8093 | MuJoCo viewer MJPEG capture |
| `go2_api.py` | 8094 | Go2 robot REST API (stand/crouch) |

All accept `--help` for options. All are started/stopped by `run.sh` / `stop.sh`.

---

## File map

| File | Location | Purpose |
|------|----------|---------|
| `inference.py` | Container | Core YOLO detection loop with thermal management, disk guards, output rotation |
| `stream_server.py` | Container | MJPEG + TCP metadata server for remote viewing |
| `stream_client.py` | Host | Measures video FPS, metadata rate, latency from operator side |
| `lidar_stream.py` | Host | LD19 LiDAR serial reader, broadcasts 360-degree scans on TCP :8092 |
| `dashboard_server.py` | Host | FastAPI + WebSocket Mission Control dashboard (v2) |
| `sim_capture.py` | Host | MuJoCo viewer screen capture → MJPEG on :8093 |
| `go2_api.py` | Host | Go2 control REST API wrapping SDK2 / go2_cmd.py |
| `network_test.py` | Host | Orchestrates network degradation experiments (6 profiles, userspace proxy) |
| `analyse_results.py` | Host | Generates 6 thesis-quality plots from experiment data |
| `status.py` | Host | Quick terminal dashboard: container, GPU temp, disk, process status |
| `run.sh` | Host | Universal launcher (inference / stream / experiment / dashboard / sim / …) |
| `stop.sh` | Host | Kills all pipeline processes cleanly |
| `yolo-inference.service` | systemd | Auto-starts headless inference on boot |
| `lidar_test.py` | Host | Low-level LD19 packet decode reference |

---

## Network experiment

The Tegra kernel (5.15.136-tegra) lacks `sch_netem`, so network emulation is implemented as a **userspace proxy** (`ConstrainedProxy`) that applies delay, bandwidth limiting, and packet loss per-connection. This is deterministic and individually measurable — preferable for thesis experiments.

### Profiles

| # | Profile | Delay | Bandwidth | Loss |
|---|---------|-------|-----------|------|
| 1 | Baseline | 0 ms | unlimited | 0% |
| 2 | Good 5G | 20 ms | 50 Mbps | 0% |
| 3 | Congested 5G | 50 ms | 10 Mbps | 0% |
| 4 | Poor 5G | 100 ms | 5 Mbps | 1% |
| 5 | Satellite NTN | 600 ms | 1 Mbps | 2% |
| 6 | Degraded NTN | 1200 ms | 0.5 Mbps | 5% |

### Running

```bash
# Full experiment (recommended)
~/Desktop/run.sh --experiment

# Quick 2-profile test
python3 ~/Desktop/network_test.py --duration 10 --profiles 1_baseline,6_degraded_ntn

# Generate plots from existing data
python3 ~/Desktop/analyse_results.py --input network_results/full_experiment_v2/summary.csv

# Live simulation from dashboard (no time limit, stop manually)
# → Use Network Simulation buttons in the dashboard Controls panel
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
| GPU < 70°C | Green overlay, normal operation |
| GPU 70–85°C | Yellow overlay, normal operation |
| GPU > 85°C | Red overlay, skip every other frame (throttle) |
| GPU > 95°C | Hard halt — inference stops |

Configurable via `--thermal-throttle` and `--thermal-halt`.

---

## Target classes

Post-inference filter (no retraining needed to change):

`person`, `dog`, `cat`, `car`, `bicycle`, `motorcycle`, `truck`, `bus`

Modify `RELEVANT_CLASSES` in `inference.py` to add/remove classes.

---

## Requirements

See [`requirements.md`](requirements.md) for formal requirements across 10 categories. See [`deployment_checklist.md`](deployment_checklist.md) for Go2 physical deployment steps.

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
