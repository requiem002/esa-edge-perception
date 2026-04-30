# CV Pipeline Requirements — Network-Adaptive Edge-AI Perception for Robotic Operations

**Platform:** Unitree Go2 / NVIDIA Jetson AGX Orin / ESA ECSAT 5G/6G Hub
**Revision:** 0.3 (2026-04-30) — derived from `requirements.xlsx` Rev 0.2, statuses updated to reflect implemented work.

Status legend: **Done** · **Partial** · **TODO** · **Phase 2**

---

## 1. Object Detection

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| DET-1 | Detect target classes in real time: person, dog, cat, car, bicycle, motorcycle, truck, bus | Must | 1 | **Done** | Post-inference filter via `RELEVANT_CLASSES` |
| DET-2 | ≥ 0.50 mAP@0.5 on target classes (dawn/daylight/dusk) | Must | 1 | **Partial** | Pretrained yolo26n meets this on COCO val; no labelled field dataset yet |
| DET-3 | ≥ 15 FPS on single 640×480 stream via TensorRT | Must | 1 | **Done** | 25–30 FPS sustained |
| DET-4 | ≥ 25 FPS to allow headroom | Should | 1 | **Done** | Measured 25.4 FPS sustained |
| DET-5 | Confidence threshold configurable at launch (default 0.25) | Must | 1 | **Done** | `--conf` |
| DET-6 | Add/remove target classes without retraining (post-inference filter) | Should | 1 | **Done** | `RELEVANT_CLASSES` set |
| DET-7 | Log every detection (class, conf, bbox, timestamp, frame) to CSV | Must | 1 | **Done** | `detections_*.csv` |

## 2. Thermal Management

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| THM-1 | Read GPU temp from sysfs ≥ every 5 s | Must | 1 | **Done** | Every 150 frames (~5 s @ 30 FPS) |
| THM-2 | Display GPU temp overlay when display is active | Must | 1 | **Done** | Colour-coded: green < 70 / yellow < 85 / red ≥ 85 |
| THM-3 | Throttle inference when GPU > 85 °C | Must | 1 | **Done** | Skips every other frame above `--thermal-throttle` (default 85) |
| THM-4 | Halt inference when GPU > 95 °C | Must | 1 | **Done** | Hard break at `--thermal-halt` (default 95) |
| THM-5 | Log temperature readings to CSV / telemetry file | Should | 1 | **TODO** | Currently overlay-only |
| THM-6 | Throttle/halt thresholds configurable via CLI | Should | 1 | **Done** | `--thermal-throttle`, `--thermal-halt` |

## 3. Network Resilience

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| NET-1 | Operate fully autonomously when link is unavailable (edge-only mode) | Must | 1 | **Done** | Pipeline is entirely edge-local |
| NET-2 | Stream detection results and annotated frames to remote operator when link available | Should | 2 | **Done** | `stream_server.py` (MJPEG + TCP metadata) |
| NET-3 | When link degrades (>300 ms RTT or <1 Mbps), reduce stream quality rather than drop | Should | 2 | **TODO** | Adaptive quality is RL-agent territory; the experiment quantifies what *needs* adapting |
| NET-4 | Buffer detection telemetry locally during outages; transmit on reconnect | Must | 2 | **Partial** | Local CSV serves as offline buffer; no replay-on-reconnect yet |
| NET-5 | No cloud dependency for inference | Must | 1 | **Done** | All inference local |
| NET-6 | Measure & log perception degradation under simulated 5G/NTN constraints | Must | 1 | **Done** | `network_test.py` + 6-profile userspace proxy. ESA Hub not used (supervisor's call: out-of-scope) — Tegra kernel lacks `sch_netem` so we built a deterministic userspace proxy instead. Latest run: `network_results/full_experiment_v2/` (3 streams × 6 profiles × 3 repeats × 60 s) |
| NET-7 | Differentiate degradation between high-bandwidth (video), medium (LiDAR), low (metadata) streams | Must | 1 | **Done** | New requirement from this work. Three-tier curve in `delivery_vs_profile.png`: video collapses 28→0.6 FPS, LiDAR 10→1.8 Hz, metadata stays 27 msg/s |

## 4. Camera Fault Tolerance

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| CAM-1 | Detect camera disconnection and auto-reconnect (exponential backoff, max 30 s) | Must | 1 | **Done** | `open_camera()` retry loop |
| CAM-2 | Log each disconnect/reconnect event with timestamp | Must | 1 | **Done** | Recovery counter printed |
| CAM-3 | Continue running during outages; resume automatically | Must | 1 | **Done** | Verified live during health check |
| CAM-4 | Switch between camera sources at runtime / via config | Should | 1 | **Done** | `--source` |
| CAM-5 | Support video file or RTSP stream as alternative input | Should | 1 | **Done** | Video files via `--source path.mp4`; RTSP not formally tested but `cv2.VideoCapture` supports it |

## 5. Data Logging & Storage

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| LOG-1 | Timestamped CSV log + optional video to configurable output dir | Must | 1 | **Done** | `detect_*.avi` + `detections_*.csv` |
| LOG-2 | Output rotation: keep ≤ 10 sessions, delete oldest | Must | 1 | **Done** | `rotate_output_files()` |
| LOG-3 | Stop video recording when free disk < 200 MB; CSV continues | Must | 1 | **Done** | Disk guard every 300 frames |
| LOG-4 | Output files world-readable/writable for host access | Must | 1 | **Done** | `make_world_writable()` |
| LOG-5 | Compress output video to H.264 MP4 on session end | Could | 1 | **TODO** | Currently XVID AVI |
| LOG-6 | Retention count and disk threshold configurable via CLI | Should | 1 | **TODO** | Hardcoded constants |

## 6. Latency Budgets

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| LAT-1 | Frame capture → detection overlay ≤ 100 ms (TensorRT) | Must | 1 | **Done** | ~25 ms inference + ~2 ms overlay |
| LAT-2 | Detection → CSV write ≤ 5 ms per frame | Must | 1 | **Done** | Negligible (~0.1 ms) |
| LAT-3 | Frame capture → remote operator ≤ 250 ms on healthy 5G | Should | 2 | **Done** | Measured: 21 ms (Good 5G), 77 ms (Congested 5G), 154 ms (Poor 5G) — all within budget |
| LAT-4 | Frame capture → remote operator ≤ 1000 ms on degraded NTN/satellite | Must | 2 | **Partial** | Satellite NTN: 932 ms (within budget). Degraded NTN: 1927 ms (exceeds budget — adaptive quality required, see NET-3) |
| LAT-5 | TensorRT engine deserialisation ≤ 30 s on startup | Must | 1 | **Done** | ~2 s |
| LAT-6 | Camera reconnect after transient disconnect ≤ 60 s | Must | 1 | **Done** | Backoff caps at 30 s per retry |

## 7. Robot Integration

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| INT-1 | Detection results via documented interface (shared mem / socket / ROS2 topic) | Must | 2 | **Partial** | TCP metadata interface works (`stream_server.py` :8091, length-prefixed JSON). ROS2 / OpenClaw bindings TODO |
| INT-2 | Publish detections in Unitree SDK2 / OpenClaw-compatible format | Should | 2 | **TODO** | Depends on INT-1 final interface |
| INT-3 | Run inside Docker (yolo-saad / yolo-orin:v2) with GPU via `--runtime nvidia` | Must | 1 | **Done** | Container committed; image v2 includes thermal/streaming code |
| INT-4 | Accept MuJoCo simulator camera as input | Should | 1 | **Done** | v4l2loopback `/dev/video2` + `sim_camera_feed.py` (in `~/unitree_mujoco/`) |
| INT-5 | Startable as systemd service | Must | 1 | **Done** | `yolo-inference.service` installed and `enable`d |
| INT-6 | Handle SIGINT/SIGTERM gracefully | Must | 1 | **Done** | Signal handlers registered |
| INT-7 | Safety-critical detection (person within 2 m) triggers immediate notification | Should | 2 | **TODO** | Needs depth (LiDAR, stereo, or monocular depth model). LD19 LiDAR now wired in (`lidar_stream.py`) — first step toward fusion |

## 8. Model Management

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| MDL-1 | Support .pt and .engine with auto-fallback | Must | 1 | **Done** | `resolve_model()` |
| MDL-2 | TensorRT engines built on target Jetson | Must | 1 | **Done** | Documented in CLAUDE.md |
| MDL-3 | Swap models at startup via CLI | Must | 1 | **Done** | `--model` |
| MDL-4 | Model files not committed | Must | 1 | **Done** | `.gitignore` excludes `*.pt/onnx/engine` |
| MDL-5 | Fine-tuning must freeze backbone (lesson from KITTI run) | Must | 1 | **TODO** | Re-execution pending; current `best.pt`/`best2.pt` are the broken ones |

## 9. Simulator Testing

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| SIM-1 | Accept MuJoCo frames via v4l2loopback / shared mem / network | Should | 1 | **Done** | v4l2loopback at `/dev/video2`, fed by `sim_camera_feed.py` |
| SIM-2 | Sim integration must not modify core inference loop | Must | 1 | **Done** | `--source` abstracts input |
| SIM-3 | Simulated camera intrinsics approximate physical Go2 head camera | Should | 1 | **Partial** | `<camera>` element added to Go2 MJCF (`fovy=86`, 480×640); pose validated against datasheet, not yet calibrated against real Go2 |

## 10. Multi-Stream Edge AI (new — derived from network experiment)

| ID | Requirement | Priority | Phase | Status | Notes |
|----|-------------|----------|-------|--------|-------|
| MUL-1 | Decouple high-bandwidth pixels from low-bandwidth structured outputs | Must | 1 | **Done** | Server emits MJPEG (~330 KiB/s) and JSON detections (~5 KiB/s) on separate ports — the **edge-AI advantage**. Quantified in `fps_vs_profile.png`: video collapses 28→0.6 FPS while metadata stays at 27 msg/s under degraded NTN |
| MUL-2 | Provide LiDAR as a third stream for downstream depth/safety reasoning | Must | 1 | **Done** | `lidar_stream.py` reads LD19 over `/dev/ttyTHS1` (host) and broadcasts 360° scans on TCP :8092 |
| MUL-3 | Three-tier degradation observable: video first, LiDAR second, metadata last | Must | 1 | **Done** | See `delivery_vs_profile.png` and `throughput_vs_profile.png`. Justifies the design: under bandwidth pressure, downgrade to LiDAR + metadata while preserving control |
| MUL-4 | Server-side log enables sent-vs-received reconciliation per profile | Should | 1 | **Done** | `stream_server.py --log server_log.csv`, `lidar_stream.py --log lidar_server_log.csv`. Plotted in `server_vs_client.png` |

---

## Status summary (Rev 0.3 — 2026-04-30)

| Category | Total | Done | Partial | TODO | Phase 2 |
|----------|------:|-----:|--------:|-----:|--------:|
| Detection      | 7  | 6  | 1 | 0 | 0 |
| Thermal Mgmt   | 6  | 5  | 0 | 1 | 0 |
| Network        | 7  | 5  | 1 | 1 | 0 |
| Camera         | 5  | 5  | 0 | 0 | 0 |
| Logging        | 6  | 4  | 0 | 2 | 0 |
| Latency        | 6  | 5  | 1 | 0 | 0 |
| Integration    | 7  | 4  | 1 | 2 | 0 |
| Model Mgmt     | 5  | 4  | 0 | 1 | 0 |
| Simulator      | 3  | 2  | 1 | 0 | 0 |
| Multi-Stream   | 4  | 4  | 0 | 0 | 0 |
| **TOTAL**      | **56** | **44** | **5** | **7** | **0** |

(Δ vs Rev 0.2: +5 new requirements from this work; Done count up from 30 → 44.)
