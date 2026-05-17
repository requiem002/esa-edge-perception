# ESA Edge-Perception Mission Control Dashboard
## Requirements Document — for Claude Code
**Version:** 1.0  
**Prepared for:** Claude Sonnet (Claude Code session)  
**Context:** This document is self-contained. Read it fully before touching any file.

---

## 1. Executive Summary

Build a browser-based real-time mission control dashboard for the ESA Edge-Perception robot dog system running on a Unitree Go2 + NVIDIA Jetson AGX Orin. The dashboard aggregates all three live data streams (MJPEG video, YOLO detection metadata, LD19 LiDAR scans), provides operator controls, and serves over Tailscale or local network so any device can access it.

The deliverable is **two new files** and a **modification to `run.sh`**:
1. `~/Desktop/dashboard_server.py` — FastAPI + WebSocket bridge (runs on HOST, not inside Docker)
2. `~/Desktop/dashboard.html` — single-file frontend (served by the backend)
3. `~/Desktop/run.sh` — add `--dashboard` flag

**Aesthetic:** NASA mission-control meets terminal. Dark charcoal background, phosphor green for live telemetry, amber for warnings/alerts, muted blue for structural chrome. Monospace font throughout (JetBrains Mono or fallback to `Courier New`). Subtle scanline CSS overlay for texture. Data-dense layout — every pixel earns its place.

---

## 2. System Context (Read This Before Writing Any Code)

### 2.1 Existing Architecture

```
[Jetson Host OS]
  ├── lidar_stream.py          ← reads /dev/ttyTHS1, broadcasts on TCP :8092
  ├── run.sh / stop.sh         ← process orchestrator
  └── [Docker: yolo-saad]
        ├── stream_server.py   ← MJPEG on HTTP :8090, metadata on TCP :8091
        └── inference.py       ← YOLO TensorRT inference loop

[Operator Device / Browser]
  └── dashboard (NEW) ← connects to :8080 via HTTP + WebSocket
```

### 2.2 Existing Ports — DO NOT CONFLICT

| Port | Protocol | Owner | Content |
|------|----------|-------|---------|
| 8090 | HTTP (MJPEG) | Docker: stream_server.py | Annotated video frames, multipart |
| 8091 | TCP (raw) | Docker: stream_server.py | JSON detection metadata, length-prefixed |
| 8092 | TCP (raw) | Host: lidar_stream.py | JSON LiDAR scans, length-prefixed |
| **8080** | **HTTP + WS** | **NEW: dashboard_server.py** | Dashboard HTML + WebSocket bridge |

### 2.3 Critical Protocol Detail — TCP Length Framing

Both TCP streams (8091 and 8092) use **4-byte big-endian length prefix** followed by UTF-8 JSON. Browsers cannot connect to raw TCP. The dashboard backend MUST bridge these to WebSocket.

```python
# Reading a length-prefixed message (existing protocol):
header = recv_exactly(sock, 4)
length = struct.unpack(">I", header)[0]
payload = recv_exactly(sock, length)
msg = json.loads(payload)
```

### 2.4 Existing Data Schemas

#### Port 8091 — Detection Metadata (from stream_server.py)
```json
{
  "frame_number": 42,
  "send_timestamp": 1714000000.123,
  "inference_time_ms": 10.2,
  "detections": [
    {"class": "person", "confidence": 0.87, "bbox": [120.1, 45.3, 310.7, 480.0]},
    {"class": "car",    "confidence": 0.91, "bbox": [0.0, 200.0, 150.0, 420.0]}
  ],
  "fps": 28.4,
  "jpeg_bytes": 41230
}
```

#### Port 8092 — LiDAR Scan (from lidar_stream.py)
```json
{
  "scan_id": 147,
  "send_timestamp": 1714000000.456,
  "rotation_speed_deg_s": 4500,
  "num_points": 432,
  "points": [
    [0.0,   1240, 200],
    [0.83,  1238, 198],
    [359.2, 800,  210]
  ]
}
```
`points` format: `[angle_degrees (0-360), distance_mm, confidence (0-255)]`

**Note:** The LD19 produces full 360 degree scans at ~10 Hz (not just one direction). Each scan has ~400+ points. `distance_mm = 0` means invalid/no return — filter these out in the frontend renderer.

### 2.5 GPU Temperature — sysfs path

The GPU temperature is readable from the Jetson HOST (not inside Docker):
```python
# Scan thermal zones for 'gpu-thermal' type
base = Path("/sys/devices/virtual/thermal/")
for zone in sorted(base.glob("thermal_zone*")):
    if (zone / "type").read_text().strip() == "gpu-thermal":
        return int((zone / "temp").read_text().strip()) / 1000.0  # Celsius
```
The dashboard backend should poll this every 2 seconds and include it in the WebSocket push.

---

## 3. Architecture Decision

### Why a sidecar backend (not pure frontend, not extending stream_server.py)

- **Browsers cannot speak raw TCP.** The metadata (8091) and LiDAR (8092) streams use raw TCP with length-prefix framing. A WebSocket bridge is mandatory.
- **stream_server.py runs inside Docker** and should not be modified — it is the subject of the network experiments. Coupling the dashboard to it risks contaminating experiment results.
- **Host-level controls** (camera toggle, LiDAR toggle, logging toggle) require subprocess/signal management at the host level. Only a host-side backend can do this cleanly.
- **FastAPI** is the right choice: async from the ground up, built-in WebSocket support, serves static files, minimal dependencies.

### Backend role (dashboard_server.py)
1. Serve `dashboard.html` at `GET /`
2. WebSocket endpoint at `ws://<host>:8080/ws` — push unified telemetry JSON to all connected browsers
3. Internally subscribe to 8091 and 8092 via asyncio TCP clients; reconnect automatically if streams drop
4. Poll GPU temp every 2 seconds
5. Track per-stream FPS/Hz by counting messages over a sliding window
6. REST control endpoints: `POST /control/{resource}/{action}`
7. Process lifecycle management for camera (stream_server.py in Docker) and LiDAR (lidar_stream.py on host)

---

## 4. File Deliverables

### 4.1 `~/Desktop/dashboard_server.py`

Single Python file. Runs on HOST (Ubuntu 22.04, Python 3.10+). All logic in one file for simplicity.

**Dependencies to install:**
```bash
pip3 install fastapi uvicorn websockets httpx
```

**Must implement:**

#### Startup
```
usage: python3 dashboard_server.py [--host 0.0.0.0] [--port 8080]
                                    [--video-host 127.0.0.1]
                                    [--meta-host 127.0.0.1]  [--meta-port 8091]
                                    [--lidar-host 127.0.0.1] [--lidar-port 8092]
                                    [--mock]
```
All defaults assume Jetson-local operation (127.0.0.1). When accessed remotely over Tailscale, the operator opens the dashboard on port 8080 but the backend (running on the Jetson) still connects locally to 8091 and 8092.

#### HTTP Routes
```
GET  /              -> serves dashboard.html (embedded as a string in the Python file)
GET  /health        -> {"status": "ok", "uptime_s": float}
GET  /video         -> proxy stream from http://127.0.0.1:8090/stream (fixes cross-origin)
POST /control/camera/start   -> starts stream_server.py in Docker
POST /control/camera/stop    -> stops it
POST /control/lidar/start    -> starts lidar_stream.py subprocess
POST /control/lidar/stop     -> stops it
POST /control/logging/start  -> enables CSV logging on both streams (restart with --log flag)
POST /control/logging/stop   -> disables logging
GET  /control/status         -> {"camera": "running"|"stopped", "lidar": "running"|"stopped", "logging": bool}
```

#### WebSocket Endpoint: `ws://<host>:8080/ws`

Push a unified telemetry JSON message every time new data arrives (target: ~10 Hz cadence matching LiDAR). Structure:

```json
{
  "type": "telemetry",
  "timestamp": 1714000000.789,
  "camera": {
    "active": true,
    "fps": 28.4,
    "inference_ms": 10.2,
    "frame_number": 4200,
    "jpeg_bytes": 41230,
    "last_seen_s": 0.05
  },
  "detections": {
    "count": 2,
    "classes": {"person": 1, "car": 1},
    "items": [
      {"class": "person", "confidence": 0.87},
      {"class": "car", "confidence": 0.91}
    ]
  },
  "lidar": {
    "active": true,
    "hz": 9.8,
    "scan_id": 147,
    "num_points": 432,
    "points": [[0.0, 1240, 200]],
    "forward_distance_mm": 1240,
    "last_seen_s": 0.1
  },
  "system": {
    "gpu_temp_c": 54.2,
    "gpu_status": "nominal",
    "uptime_s": 3610.4
  },
  "streams": {
    "video_alive": true,
    "meta_alive": true,
    "lidar_alive": true
  }
}
```

`gpu_status` values: `"nominal"` (< 70C), `"warm"` (70-85C), `"hot"` (> 85C), `"critical"` (> 95C).  
`last_seen_s` is seconds since last message received on that stream — use this for connection health dots.

#### Stream Connection Strategy
- Use `asyncio` throughout. Do not use threads for the TCP readers.
- TCP readers: reconnect with exponential backoff (0.5s, 1s, 2s, 4s, max 8s) if connection drops.
- If a stream is not alive, set `*_alive: false` and zero/null the relevant telemetry fields — do not crash.
- Multiple WS clients must be supported (the browser may be open on multiple devices).

#### Process Management
- Camera: `docker exec yolo-saad python3 /workspace/stream_server.py` — check if container is running first with `docker inspect`.
- LiDAR: `python3 ~/Desktop/lidar_stream.py` — managed as asyncio subprocess.
- Store subprocess handles; on stop, send SIGTERM then wait up to 5s, then SIGKILL.
- If the stream processes were started externally (e.g. via run.sh), the dashboard should detect them as already running via port probe rather than refusing to work.

#### Port Detection (startup resilience)
On startup, attempt to connect to 8091 and 8092. If they're not up yet, start retrying. Do not require the streams to be running before the dashboard starts — the operator should be able to start streams from the dashboard UI itself.

#### Mock Mode (`--mock` flag)
When `--mock` is passed:
- Generate synthetic telemetry: random detections, simulated LiDAR scan (concentric rings + noise), FPS oscillating between 0.6 and 28.5
- Do not attempt TCP connections to 8091/8092
- GPU temp oscillates 50-90C cycling through all thermal states
- All control endpoints return 200 OK without spawning any processes
- This enables frontend development and testing without a Jetson

---

### 4.2 `~/Desktop/dashboard.html`

Single self-contained HTML file. All CSS and JS inline. No external CDN dependencies (works air-gapped). JetBrains Mono font via Google Fonts @import with graceful fallback to `monospace`.

The dashboard.html file is **embedded as a Python string inside dashboard_server.py** and served at `GET /`. This keeps deployment to a single file.

**Target resolution:** 1920x1080 (full-screen browser). Must not scroll — everything fits in one viewport.

#### Layout (CSS Grid)

```
┌─────────────────────────────────┬──────────────────┬──────────────┐
│  TOP STATUS BAR (full width, ~28px)                               │
├─────────────────────────────────┼──────────────────┼──────────────┤
│                                 │                  │   CONTROLS   │
│         CAMERA FEED             │   LIDAR POLAR    │   PANEL      │
│         (MJPEG stream)          │   SWEEP          │              │
│         ~55% width              │   ~25% width     │   ~20% width │
│         ~60% height             │   ~60% height    │   ~60% height│
├────────────────┬────────────────┴──────────────────┴──────────────┤
│  DETECTIONS    │         TELEMETRY STRIP                          │
│  (class list   │  FPS | LiDAR Hz | Inference | Health | GPU Temp  │
│  + counts)     │                                                  │
│  ~25% width    │  ~75% width                                      │
│  ~40% height   │  ~40% height                                     │
└────────────────┴──────────────────────────────────────────────────┘
```

#### Colour Palette (CSS variables)

```css
:root {
  --bg-primary:    #0d0f0e;
  --bg-panel:      #111814;
  --bg-panel-alt:  #0a0f0c;
  --border:        #1e3028;
  --green-bright:  #00ff88;
  --green-dim:     #00aa55;
  --green-dark:    #005522;
  --amber:         #ffaa00;
  --amber-dark:    #aa6600;
  --red-alert:     #ff3333;
  --blue-accent:   #0066cc;
  --text-primary:  #c8ffd4;
  --text-dim:      #4a7a5a;
  --text-muted:    #2a4a35;
}
```

#### Typography
- All text: `font-family: 'JetBrains Mono', 'Courier New', monospace`
- Panel headers: uppercase, `letter-spacing: 0.15em`, `font-size: 0.65rem`, colour `var(--text-dim)`
- Live values: larger, bright green, `font-variant-numeric: tabular-nums`
- Labels: dim green

#### Scanline Overlay
```css
body::after {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 9999;
  background: repeating-linear-gradient(
    to bottom,
    rgba(0,20,10,0.12) 0px,
    rgba(0,20,10,0.12) 1px,
    transparent 1px,
    transparent 3px
  );
}
```

#### Panel Styling
- All panels: `border: 1px solid var(--border)`, `background: var(--bg-panel)`
- Panel header bar: 22px tall, `background: var(--bg-panel-alt)`, `border-bottom: 1px solid var(--border)`
- Header text format example: `[ ◈ CAMERA FEED ]` — small unicode glyphs for retro feel
- Decorative corner brackets via CSS pseudo-elements in dim green

---

### 4.3 Panel Specifications

#### Panel A — Camera Feed
- `<img id="videoFeed">` with `src="/video"` (proxied through dashboard backend to avoid CORS)
- On `onerror`: show grey placeholder div with `[ NO SIGNAL ]` in amber, blinking 1Hz via CSS animation
- Blinking `■ LIVE` indicator in top-right corner of the feed
- Detection count overlay bottom-left from WS telemetry: `DET: 002`
- Current FPS from WS telemetry bottom-right: `28.4 FPS`

#### Panel B — LiDAR Polar Sweep
- Canvas element, square aspect ratio, fills panel
- Render full 360 degree polar plot: angle on circumference, distance as radius
- Filter: skip points where `distance_mm == 0` or `confidence < 50`
- Max display range: 5000mm default (5m) — rings labeled in metres
- Concentric rings at 25%, 50%, 75%, 100% of max range — dim green dashed lines
- Cardinal direction labels: `FWD` (top, 0 deg), `R` (90 deg), `AFT` (180 deg), `L` (270 deg)
- Point colour gradient by distance: `#00ff88` (< 1m) -> `#ffaa00` (2.5m) -> `#ff3333` (5m)
- Point size: 2px radius filled circles
- Persistence: last 3 scans, older scans at reduced opacity (0.3, 0.15)
- Forward arc highlight: 350-10 degree wedge with subtle bright green fill
- Centre marker: small robot icon or `[G2]` text label
- Rotating sweep line: thin green line animated at `rotation_speed_deg_s` from telemetry
- Panel header shows: `SCAN #147 | 9.8 Hz | 432 pts`

#### Panel C — Controls
Layout as a vertical panel, top-right:
```
[ ◈ OPERATOR CONTROLS ]
━━━━━━━━━━━━━━━━━━━━━━━
CAMERA STREAM
  ● RUNNING   [■ STOP]

LIDAR SENSOR
  ● RUNNING   [■ STOP]

DATA LOGGING
  ○ INACTIVE  [▶ START]

━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM
  ● yolo-saad  ACTIVE
  ↑ 01:00:34 uptime

━━━━━━━━━━━━━━━━━━━━━━━
  [⚠ EMERGENCY STOP]
```
- Toggle buttons: `border: 1px solid var(--green-dim)`, no border-radius, monospace text
- Active indicator: `●` in green; inactive: `○` in dim green
- Emergency Stop: red border, amber text, calls all three stop endpoints simultaneously
- Status indicators update from WS telemetry only, not optimistic button-press updates

#### Panel D — Detections
```
[ ◈ DETECTIONS ]
━━━━━━━━━━━━━━━
COUNT:  002
FRAME:  04200

  PERSON  ████░  0.87
  CAR     █████  0.91

  ── SESSION TOTALS ──
  person:    2847
  car:        341
  bicycle:      4
```
- Confidence bar uses unicode block chars (`█`) scaled to confidence (0.0-1.0 = 0-5 blocks)
- Confidence colour: `var(--green-bright)` > 0.7, `var(--amber)` 0.4-0.7, `var(--red-alert)` < 0.4
- Session totals accumulated locally from WS messages (JS variable, resets on page reload)
- Maximum 8 classes displayed, sorted by confidence descending

#### Panel E — Telemetry Strip (bottom, wide, 5 sub-panels inline)

**Sub-panel 1: Video FPS**
- Large numeric: `28.4` in `var(--green-bright)` or colour-coded
- Label: `VIDEO FPS`
- Mini sparkline canvas below: last 60 values, 120x30px canvas
- Threshold line at 15 FPS drawn in amber dashes on sparkline
- Colour: green >= 15, amber 5-15, red < 5

**Sub-panel 2: LiDAR Hz**
- Large numeric: `9.8`
- Label: `LIDAR Hz`
- Sparkline: last 60 values
- Colour: green >= 5, amber 2-5, red < 2

**Sub-panel 3: Inference Latency**
- Display: `10.2 ms`
- Label: `INFER. LATENCY`
- Colour: green < 20ms, amber 20-50ms, red > 50ms

**Sub-panel 4: Stream Health**
Three rows with coloured dot + label:
```
● VIDEO STREAM
● METADATA
● LIDAR
```
Dot colours based on `last_seen_s`:
- Green: < 2.0s (alive)
- Amber: 2.0-5.0s (stale)
- Red: > 5.0s or `active: false` (dead)

**Sub-panel 5: GPU Temperature**
```
GPU TEMP
54.2 °C
[NOMINAL]
```
- Colour-coded: green NOMINAL (<70), amber WARM (70-85), red HOT (85-95), blinking red CRITICAL (>95)

---

### 4.4 Top Status Bar

Full-width bar, ~28px tall, across the very top:
```
FENRIR-1 MISSION CONTROL  |  JETSON AGX ORIN  |  2026-05-06  14:32:07 UTC  |  GPU: 54.2°C [NOMINAL]  |  ● 3 STREAMS ACTIVE
```
- Background: `var(--bg-panel-alt)`
- Separators: `|` in dim green
- Stream count coloured: all green if 3/3, amber if 1-2/3, red if 0/3
- Clock updates every second via `setInterval`

---

### 4.5 `~/Desktop/run.sh` Modification

Add a `--dashboard` flag. Read the existing run.sh first and add the case cleanly into the existing argument parsing pattern. Also update `stop.sh` to kill dashboard if running.

```bash
# In run.sh:
--dashboard)
    echo "[RUN] Starting mission control dashboard on :8080..."
    pip3 install -q fastapi uvicorn websockets httpx 2>/dev/null
    nohup python3 ~/Desktop/dashboard_server.py > ~/Desktop/dashboard.log 2>&1 &
    echo $! > /tmp/dashboard.pid
    sleep 1
    JETSON_IP=$(hostname -I | awk '{print $1}')
    echo "[RUN] Dashboard: http://${JETSON_IP}:8080"
    ;;
```

```bash
# In stop.sh — add:
if [ -f /tmp/dashboard.pid ]; then
    kill $(cat /tmp/dashboard.pid) 2>/dev/null
    rm /tmp/dashboard.pid
    echo "[STOP] Dashboard stopped"
fi
```

---

## 5. User Stories

| ID | As a... | I want to... | So that... |
|----|---------|--------------|------------|
| US-01 | Operator | See the live annotated camera feed in the browser | I can observe what the robot sees from any device on the network |
| US-02 | Operator | See the LiDAR sweep as a full 360 degree polar plot | I can monitor the robot's spatial environment |
| US-03 | Operator | See current detections with class and confidence | I can verify the YOLO pipeline is working correctly |
| US-04 | Operator | See video FPS with the 15 FPS threshold visually marked | I can immediately tell when the network is degraded |
| US-05 | Operator | See GPU temperature colour-coded by severity | I know if the Jetson is approaching a thermal limit |
| US-06 | Operator | Toggle the camera stream on/off from the UI | I can stop the stream without SSH access |
| US-07 | Operator | Toggle the LiDAR on/off from the UI | I can conserve bandwidth or restart a hung sensor |
| US-08 | Operator | Toggle data logging on/off from the UI | I can start a logging session without SSH |
| US-09 | Operator | Hit an Emergency Stop button | I can halt all streams immediately |
| US-10 | Researcher | See per-stream health indicators (video / metadata / LiDAR) | I can distinguish stream-specific failures from network-wide failures |
| US-11 | Researcher | See inference latency in milliseconds live | I can monitor TensorRT performance during experiments |
| US-12 | Operator | Access the dashboard from any Tailscale-connected device | I am not restricted to the Jetson's local display |
| US-13 | Operator | See session-cumulative detection counts by class | I can understand what the robot has observed over a run |
| US-14 | Operator | See the dashboard still function if one stream is down | Partial failures do not black out the whole dashboard |

---

## 6. Tech Stack Constraints

### Backend (dashboard_server.py)

| Constraint | Requirement |
|------------|-------------|
| Runtime | Python 3.10+ on Ubuntu 22.04 (Jetson host OS) |
| Framework | FastAPI + Uvicorn |
| Install | `pip3 install fastapi uvicorn websockets httpx` |
| NOT inside Docker | Runs on HOST only |
| No ROS | No ROS 2, no rospy, no ROS topics whatsoever |
| Async | Use `asyncio` for all IO. No `threading` for network IO. |
| Single file | All backend logic in `dashboard_server.py` — no sub-modules |
| No database | All state in-memory |
| Subprocess | Use `asyncio.create_subprocess_exec` for process control |
| dashboard.html | Embedded as a triple-quoted string inside dashboard_server.py |

### Frontend (embedded in dashboard_server.py as a string)

| Constraint | Requirement |
|------------|-------------|
| Single file | All HTML, CSS, JS in one string — no external files |
| No build step | No npm, no webpack, no TypeScript |
| No CDN | All assets inline. Google Fonts via @import only (with fallback). |
| Vanilla JS only | No React, Vue, or any framework |
| Canvas API | Use Canvas 2D for LiDAR polar plot and sparklines |
| WebSocket | Native browser WebSocket API |
| MJPEG | `<img src="/video">` — video proxied through dashboard backend |

### Deployment Environment

| Item | Value |
|------|-------|
| Jetson OS | Ubuntu 22.04, Python 3.10 |
| Docker container | `yolo-saad` (image `yolo-orin:v2`), `--network host` |
| Camera | `/dev/video0` inside container |
| LiDAR serial | `/dev/ttyTHS1` on host (not accessible from container) |
| Dashboard port | `8080` |
| Video port | `8090` (HTTP MJPEG) |
| Metadata port | `8091` (TCP, length-prefixed JSON) |
| LiDAR port | `8092` (TCP, length-prefixed JSON) |
| Working directory | `~/Desktop/` — all new files go here |
| Git repo | `~/Desktop/` files are tracked in `github.com/requiem002/esa-edge-perception` |

---

## 7. Acceptance Criteria

### AC-01: Dashboard loads
- `GET http://<jetson>:8080` returns a 200 with the dashboard HTML
- No JS console errors on load
- Layout fills 1920x1080 viewport with no scrollbars

### AC-02: Video feed renders
- Camera feed appears in Panel A when stream_server.py is running
- Annotated bounding boxes visible
- If camera is stopped: `[ NO SIGNAL ]` placeholder appears (not a broken image icon)

### AC-03: LiDAR polar plot renders
- 360 degree point cloud visible on polar canvas
- Points at 0mm filtered out
- Distance rings and cardinal direction labels present
- Plot updates at ~10 Hz when lidar_stream.py is running

### AC-04: Detections panel is live
- Class names, confidence scores, and counts update within 200ms of metadata arrival
- Session totals increment correctly

### AC-05: FPS gauge shows correct value and threshold
- Video FPS within ±1 FPS of value in metadata stream
- 15 FPS threshold line visible on sparkline
- Colour changes: green >= 15, amber 5-15, red < 5

### AC-06: Stream health dots respond correctly
- Dot turns amber after 2s of no messages
- Dot turns red after 5s
- Dot returns to green immediately on message resume

### AC-07: GPU temperature updates
- Displayed and updates every ~2s
- Colour matches thresholds (NOMINAL/WARM/HOT/CRITICAL)

### AC-08: Camera toggle works end-to-end
- `[■ STOP]` sends POST /control/camera/stop
- stream_server.py Docker process terminates within 5s
- Video panel shows NO SIGNAL, metadata health dot goes red
- `[▶ START]` restarts it; video resumes

### AC-09: LiDAR toggle works end-to-end
- Same pattern as AC-08 for lidar_stream.py subprocess

### AC-10: Emergency stop halts everything
- All three stop endpoints called
- All stream health dots red within 5s

### AC-11: Auto-reconnect (backend)
- If port 8091 or 8092 goes offline and comes back, backend reconnects within 10s without restart

### AC-12: Multi-client
- Two browsers open simultaneously work without errors

### AC-13: Graceful partial failure
- If LiDAR not running: rest of dashboard functions, LiDAR panel shows `[ NO SIGNAL ]`
- If camera not running: LiDAR and telemetry still display
- No unhandled exceptions or white screens

### AC-14: Mock mode
- `python3 dashboard_server.py --mock` starts without hardware
- Synthetic telemetry flows to browser; all panels animate
- Useful for testing on a laptop without Jetson

---

## 8. Non-Goals (Explicit Scope Limits)

- No historical data persistence across sessions
- No authentication — local/Tailscale network only
- No video recording from dashboard (handled by inference.py separately)
- No network profile switching from dashboard (use run.sh --experiment)
- No 3D LiDAR — LD19 is 2D, polar plot is correct
- No WebRTC — MJPEG is sufficient
- No mobile-responsive layout — 1920x1080 desktop target
- **Do not modify `stream_server.py`** — experiment-critical, treat as read-only
- **Do not modify `lidar_stream.py`** — treat as read-only, control via subprocess only  
- **Do not modify `inference.py`** — treat as read-only

---

## 9. Implementation Order (Suggested)

1. Read all existing files: `run.sh`, `stop.sh`, `stream_server.py`, `lidar_stream.py` before writing anything
2. Write `dashboard_server.py` skeleton: FastAPI app, `/health` route, `--mock` flag skeleton
3. Add TCP client for port 8091 with asyncio reconnect loop
4. Add TCP client for port 8092 with asyncio reconnect loop
5. Add GPU temp poller (async task, 2s interval)
6. Add WebSocket `/ws` endpoint and unified telemetry push to all connected clients
7. Add control endpoints and subprocess management (camera + LiDAR + logging)
8. Add `/video` proxy route using httpx streaming
9. Write `dashboard.html` layout structure (HTML + CSS grid + colour variables + scanlines)
10. Implement top status bar
11. Implement Panel A: camera feed with NO SIGNAL fallback
12. Implement Panel B: LiDAR polar canvas renderer
13. Implement Panel C: controls panel with toggle buttons + emergency stop
14. Implement Panel D: detections panel with session totals
15. Implement Panel E: telemetry strip with sparklines + health dots + GPU temp
16. Wire frontend WebSocket client: connect, parse telemetry, update all panels
17. Embed `dashboard.html` string into `dashboard_server.py`
18. Update `run.sh` and `stop.sh`
19. Test with `--mock` flag, verify all panels animate correctly
20. Test with live Jetson streams

---

## 10. Key Implementation Notes

### Cross-origin video fix
The video stream is on port 8090 (different port = different CORS origin from port 8080). The backend provides a `/video` proxy route that streams the MJPEG from 8090 to the browser on the same origin. Frontend uses `src="/video"` not `src="http://...:8090/stream"`.

```python
# In FastAPI:
@app.get("/video")
async def proxy_video(request: Request):
    async def stream_video():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", "http://127.0.0.1:8090/stream") as r:
                async for chunk in r.aiter_bytes(chunk_size=4096):
                    yield chunk
    return StreamingResponse(stream_video(), media_type="multipart/x-mixed-replace; boundary=frame")
```

### LiDAR canvas performance
At 10 Hz with 400+ points, use `requestAnimationFrame`, not `setInterval`. Store latest scan data in a JS variable. Redraw entire canvas each frame with `ctx.clearRect()`. 400 dots at 2px radius is trivial for Canvas 2D.

### WebSocket reconnect in frontend
```javascript
let ws;
function connectWS() {
  ws = new WebSocket(`ws://${window.location.host}/ws`);
  ws.onmessage = (e) => handleTelemetry(JSON.parse(e.data));
  ws.onclose = () => { updateConnectionStatus(false); setTimeout(connectWS, 2000); };
  ws.onerror = () => ws.close();
  ws.onopen = () => updateConnectionStatus(true);
}
```

### Process detection (already running via run.sh)
Do not assume processes were started by the dashboard. On `/control/status`, probe ports to determine what is actually running, regardless of how it was started.

### The LD19 produces full 360 degree scans
`lidar_stream.py` assembles complete 360 degree rotations before broadcasting. Port 8092 delivers full scans. The polar plot will have data across the full circle, not just one direction. `lidar_test.py` is a different file used for low-level testing only — do not reference it.

### docker exec vs docker start
stream_server.py runs INSIDE the existing `yolo-saad` container. The container itself is managed by the systemd service. To start streaming: `docker exec -d yolo-saad python3 /workspace/stream_server.py`. To stop: find the PID inside the container and kill it, or use `docker exec yolo-saad pkill -f stream_server.py`.

### FPS calculation in backend
Do not rely on the `fps` field in the metadata JSON (it is the server-side inference FPS). Calculate actual delivery FPS independently in the dashboard backend by counting messages received per second using a sliding window of the last 2 seconds of timestamps.
