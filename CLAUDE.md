# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This is an NVIDIA Jetson Orin device (aarch64, Ubuntu 22.04, JetPack R36, kernel 5.15.136-tegra) used for YOLO-based object detection. Development happens inside a Docker container.

**CUDA**: `/usr/local/cuda/` (12.2) — `nvcc` and GPU tools available here.

## Docker Development Container

The working container is named `yolo-saad` (image `yolo-orin:v1`), committed from `dustynv/l4t-pytorch:r36.2.0`. Do **not** use `ultralytics/ultralytics:latest-jetson` — that image targets JetPack 5 and does not work on this device. Docker commands work **without sudo**.

```bash
docker run -it --name yolo-saad --ipc=host --runtime nvidia --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/video0 \
  -v ~/Desktop:/workspace \
  yolo-orin:v1 bash
```

Inside the container, `/workspace` maps to `~/Desktop`.

## Python Package Installation Rules

These rules exist because the Jetson requires ARM-specific wheels built against the system CUDA/cuDNN. PyPI wheels for these packages are x86 or will pull in incompatible dependencies.

- **Never** install `opencv-python` or `torch`/`torchvision` from PyPI.
- Always install with `--no-deps` when adding packages that could pull in torch or opencv as transitive dependencies.
- If a package is unavailable on the default index, use the Jetson pip index as a fallback: `https://pypi.jetson-ai-lab.dev/jp6/cu122`

## Model Pipeline

```
Train → best.pt  (PyTorch)
           ↓  yolo export
        best.onnx  (ONNX)
           ↓  TensorRT
        best.engine  (TensorRT, Jetson-optimized)
           ↓
        Inference on video/images
```

### Key CLI Commands

```bash
# Export PyTorch model to ONNX/TensorRT engine
yolo export model=yolo26n.pt format=onnx
yolo export model=yolo26n.pt format=engine

# Run inference
yolo predict model=yolo26n.engine source=/dev/video0  # webcam
yolo predict model=yolo26n.pt source=bus.jpg          # image

# Output lands in runs/detect/predictN/
```

## Files on Desktop

- `yolo26n.pt/onnx/engine` — **current working model** (YOLO nano)
- `best.pt`, `best.onnx`, `best.engine` — KITTI-trained model, **broken/non-functional**, currently being debugged
- `best2.pt`, `best2.onnx`, `best2.engine` — second broken variant
- `runs/detect/` — inference outputs (images, `.avi` video)

## TensorRT Samples

Reference Python scripts for manual ONNX→TensorRT conversion at:
`/usr/src/tensorrt/samples/python/yolov3_onnx/`

## Unitree Go2 Simulator

The Unitree MuJoCo simulator lives at `~/unitree_mujoco/` on the host (not inside the Docker container). It simulates a Unitree Go2 quadruped using MuJoCo physics with Unitree SDK2 integration via DDS messaging.

- **Python simulator**: `~/unitree_mujoco/simulate_python/unitree_mujoco.py`
- **C++ simulator**: `~/unitree_mujoco/simulate/` (recommended, needs `cmake && make`)
- **Robot models**: `~/unitree_mujoco/unitree_robots/go2/` (MJCF XML + meshes)
- **Config**: `~/unitree_mujoco/simulate_python/config.py` — set `DOMAIN_ID=1` for sim, `0` for real robot
- **SDK**: `~/unitree_sdk2/` (C++) and `~/unitree_sdk2_python/` (Python bindings)
- **Bridge scripts**: `~/unitree_mujoco/go2_bridge/` — control examples using SDK2

The simulator does **not** currently have camera rendering. Integrating camera output with the YOLO pipeline requires adding MuJoCo offscreen rendering (see `requirements.md` SIM-1 through SIM-3).

## Related workspaces
The Unitree simulator and OpenClaw agentic framework live in ~/esa_jetson/.
That directory has its own Claude Code instance with separate context.
Key paths: ~/unitree_mujoco/, ~/.openclaw/, ~/go2_bridge/

## OpenClaw Agentic Framework

OpenClaw (`~/.openclaw/`) is an agentic framework running on the Jetson. It has an AI agent named "Fenrir" that controls the Go2 robot through skill tools (go2_stand, go2_crouch, go2_status). The workspace is at `~/.openclaw/workspace/` and contains motion scripts (wave, stand, crouch sequences).

## Requirements

`requirements.md` on the Desktop contains formal, numbered requirements for the CV pipeline covering detection, thermal management, network resilience, camera fault tolerance, logging, latency, and robot integration. (Note: the canonical copy is `requirements.xlsx` — the `.md` mirror may not exist on disk.)

## Git repository

`~/Desktop` itself is the git repo (no file moves), tracking only source/docs.

- **Remote**: https://github.com/requiem002/esa-edge-perception
- **Branch**: `main`
- **Tracked**: `*.py`, `*.sh`, `*.md`, `*.svg`, `*.service`, `requirements.*`, `CLAUDE.md`, `architecture.svg`
- **Ignored** (`.gitignore`): models (`*.pt/onnx/engine`, `best*`, `yolo26n.*`), media (`*.avi/mp4/jpg/png`), data (`*.csv/json/xlsx`, `output/`, `runs/`, `network_results/`), local logs, desktop launchers, `__pycache__/`.

Commit and push as the regular user — `~/Desktop/.git` is owned by the user, not root.

## LIDAR (LD19 / LDLidar)

- Connected to the **host** at `/dev/ttyTHS1` at 230 400 baud (NOT inside Docker).
- Reference: `lidar_test.py` (raw packet decode example).
- Streamer: `lidar_stream.py` — reads LD19 packets, assembles full 360° scans (~10 Hz, ~300–450 points/scan), broadcasts each scan as length-prefixed JSON over TCP port **8092**.
- LD19 packet layout: `0x54 0x2C` header + 45 data bytes (12 distance/confidence triplets between `start_angle` and `end_angle`).

## Network degradation experiment

Three-stream perception pipeline measured over emulated 5G/NTN links (the Tegra kernel lacks `sch_netem`, so a userspace proxy applies delay/bandwidth/loss):

| Component        | Where        | Port |
|------------------|--------------|------|
| `stream_server.py` (MJPEG + YOLO) | inside `yolo-saad` | 8090 (video), 8091 (meta) |
| `lidar_stream.py`                 | host             | 8092 (lidar) |
| `stream_client.py`                | host             | connects via proxies on 9090/9091/9092 |
| `network_test.py`                 | host             | orchestrator (6 profiles, --repeats N) |
| `analyse_results.py`              | host             | plots + thesis table |

Runtime artefacts (all in `network_results/`, all gitignored):
- `summary.csv` — per-profile means + stddevs
- `<profile>/measurements_repN[._video|_lidar].csv` — per-repeat client traces
- `<profile>/profile.json` — config + per-repeat run windows (`run_start_ts`/`run_end_ts`)
- `server_log.csv` (`stream_server.py --log`) — every frame the server processed
- `lidar_server_log.csv` (`lidar_stream.py --log`) — every scan the LIDAR streamer published
- 6 plots: `fps_vs_profile.png`, `latency_vs_profile.png`, `delivery_vs_profile.png`, `throughput_vs_profile.png`, `quality_score.png`, `server_vs_client.png`

Latest run lives in `network_results/full_experiment_v2/` — three streams (video / metadata / lidar), 6 profiles, 60 s × 3 repeats.

To reproduce:

```bash
# 1. Start servers
docker exec -d yolo-saad python3 /workspace/stream_server.py \
    --log /workspace/network_results/full_experiment_v2/server_log.csv
nohup python3 ~/Desktop/lidar_stream.py \
    --log ~/Desktop/network_results/full_experiment_v2/lidar_server_log.csv \
    > /tmp/lidar_stream.log 2>&1 &

# 2. Run experiment
python3 ~/Desktop/network_test.py \
    --duration 60 --repeats 3 \
    --output-dir network_results/full_experiment_v2

# 3. Generate plots and thesis table
python3 ~/Desktop/analyse_results.py \
    --input network_results/full_experiment_v2/summary.csv
```

If the LIDAR isn't connected, `network_test.py` automatically detects that port 8092 is unreachable and runs camera-only.
