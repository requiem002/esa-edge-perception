# ESA Mission Control Dashboard — Enhancement Requirements v2
## For Claude Code (Sonnet)
**Prerequisite:** `dashboard_server.py` and `dashboard.html` (embedded) already exist at `~/Desktop/`.
**Read `CLAUDE.md`, `experiment_results.md`, and `lidar_stream.py` before writing any code.**
**This document is the single source of truth. Implement everything in order.**

---

## 0. Before You Start — Inspection Tasks

Before writing a single line of code, run these commands and read the output:

```bash
# 1. Understand the go2 control interface
ls ~/unitree_mujoco/go2_bridge/
cat ~/unitree_mujoco/go2_bridge/*.py 2>/dev/null | head -200

# 2. Understand OpenClaw's existing skill implementations
ls ~/.openclaw/workspace/ 2>/dev/null
find ~/.openclaw -name "*.py" | head -20 | xargs grep -l "stand\|crouch\|go2" 2>/dev/null

# 3. Understand unitree SDK2 Python interface
ls ~/unitree_sdk2_python/example/ 2>/dev/null | head -20

# 4. Find what DDS/SDK2 calls actually control stand/crouch
grep -r "stand\|SportModeState\|StandUp\|BalanceStand" ~/unitree_sdk2_python/ 2>/dev/null | head -20
grep -r "go2_stand\|go2_crouch" ~/.openclaw/ 2>/dev/null | head -20

# 5. Check if ffmpeg is installed
which ffmpeg && ffmpeg -version | head -2

# 6. Check current dashboard_server.py line count and video proxy section
wc -l ~/Desktop/dashboard_server.py
grep -n "proxy_video\|/video\|httpx" ~/Desktop/dashboard_server.py | head -20
```

Use these findings to fill in the SDK2 call details in Enhancement 4. Document what you find.

---

## 1. BUG FIX — Video Proxy Latency (Priority: High)

**Problem:** The `/video` proxy using `httpx.AsyncClient` adds ~1-3 seconds of latency because httpx buffers internally before yielding chunks. The MJPEG stream is time-sensitive.

**Fix:** Replace the httpx proxy with a raw `asyncio` socket connection that reads and yields bytes immediately with zero internal buffering.

**Implementation — replace the existing `proxy_video` route entirely:**

```python
@app.get("/video")
async def proxy_video():
    """Low-latency MJPEG proxy using raw asyncio sockets (no httpx buffering)."""
    async def stream_mjpeg():
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', 8090), timeout=3.0
            )
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return

        try:
            # Send a minimal HTTP/1.0 GET (Connection: close avoids keep-alive buffering)
            writer.write(
                b"GET /stream HTTP/1.0\r\n"
                b"Host: localhost\r\n"
                b"\r\n"
            )
            await writer.drain()

            # Skip HTTP response headers (read until double CRLF)
            header_buf = b""
            while b"\r\n\r\n" not in header_buf:
                chunk = await asyncio.wait_for(reader.read(512), timeout=5.0)
                if not chunk:
                    return
                header_buf += chunk

            # Yield any body bytes already in the buffer after headers
            sep = header_buf.find(b"\r\n\r\n") + 4
            if sep < len(header_buf):
                yield header_buf[sep:]

            # Stream remaining bytes immediately as they arrive
            while True:
                chunk = await asyncio.wait_for(reader.read(8192), timeout=10.0)
                if not chunk:
                    break
                yield chunk

        except (asyncio.TimeoutError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    return StreamingResponse(
        stream_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

**Remove the `httpx` import if it is no longer used elsewhere after this change.**

**Acceptance criterion:** Video feed visible on MacBook over Tailscale with latency < 500ms (subjectively similar to loading a live webcam feed).

---

## 2. BUG FIX — Offline State Banner (Priority: High)

**Problem:** When all three streams are dead (all `last_seen_s > 10`), the dashboard shows stale frozen values with no indication that the Jetson processes are not running. The user cannot tell if the system is offline or just degraded.

**Backend change:** Add a field to the telemetry WebSocket push:
```json
"system_state": "online" | "degraded" | "offline"
```
- `"online"`: at least one stream alive (last_seen_s < 5)
- `"degraded"`: at least one stream alive, but at least one dead
- `"offline"`: ALL streams dead (all last_seen_s > 10 or never connected)

**Frontend change:** When `system_state === "offline"`:
- Display a full-width amber banner below the status bar:
  ```
  ⚠  ALL STREAMS OFFLINE — JETSON PROCESSES NOT RUNNING  ⚠
  ```
  Styled in amber, blinking at 0.5Hz. Text is monospace, uppercase, centered.
- All panel numeric values freeze and dim to 30% opacity.
- Health dots all show red.
- The banner disappears immediately when any stream becomes alive again.

When `system_state === "degraded"`: show a smaller warning in the status bar: `DEGRADED — N/3 STREAMS ACTIVE` in amber (not a full banner).

---

## 3. ENHANCEMENT — Network Analytics Panel (Priority: High)

**Goal:** Add a new panel to the dashboard that visualises real-time network conditions derived from the existing data streams. This directly reflects the academic contribution of the project — showing the three-tier degradation hierarchy as a live visualisation.

**All metrics are computed from existing stream data. No new ports or experiment mode required.**

### 3.1 Backend — New Computed Fields in Telemetry

Add these computed fields to the WebSocket telemetry push. Compute them in the backend from a rolling 10-second window of received messages:

```json
"network": {
  "video_bw_kbps": 342.5,          // jpeg_bytes * fps / 1000 (rolling avg)
  "lidar_bw_kbps": 70.0,           // num_points * 12 bytes * hz / 1000 (approx)
  "meta_bw_kbps": 5.4,             // ~200 bytes * fps_meta / 1000
  "total_bw_kbps": 417.9,          // sum of above
  "e2e_latency_ms": 45.2,          // time.time() - send_timestamp from metadata
  "frame_loss_pct": 1.2,           // gaps in frame_number / expected frames * 100
  "quality_score": 94,             // formula below
  "inferred_profile": "Good 5G",   // nearest match to known profiles
  "profile_confidence": 0.87       // how confident we are in the match
}
```

**Quality score formula** (matches your thesis formula from experiment_results.md):
```python
fps_score = min(camera_fps / 30.0, 1.0) * 100
latency_score = max(0, 100 - e2e_latency_ms / 10)
delivery_score = (1.0 - frame_loss_pct / 100) * 100
quality_score = round(0.4 * fps_score + 0.3 * latency_score + 0.3 * delivery_score)
```

**Frame loss calculation:**
- Track the last received `frame_number` from metadata
- Count gaps: `expected_frames = frame_number_now - frame_number_10s_ago`
- Count received frames in that window
- `frame_loss_pct = (1 - received/expected) * 100` clamped to [0, 100]

**Profile inference** — map current conditions to nearest of the 6 known profiles using Euclidean distance on normalised (fps, latency_ms) space:

```python
KNOWN_PROFILES = {
    "Baseline":      {"fps": 28.4, "latency_ms": 1,    "bw_kbps": float('inf')},
    "Good 5G":       {"fps": 28.5, "latency_ms": 21,   "bw_kbps": 50000},
    "Congested 5G":  {"fps": 12.1, "latency_ms": 77,   "bw_kbps": 10000},
    "Poor 5G":       {"fps": 6.4,  "latency_ms": 152,  "bw_kbps": 5000},
    "Satellite NTN": {"fps": 1.2,  "latency_ms": 932,  "bw_kbps": 1000},
    "Degraded NTN":  {"fps": 0.6,  "latency_ms": 1927, "bw_kbps": 500},
}
```

### 3.2 Frontend — Network Panel Layout

Add a new **Panel F: Network Analytics** — insert it between the telemetry strip and the controls panel, or replace part of the bottom row. The layout adjustment is at Claude Code's discretion given the existing grid, but the panel must be visible without scrolling.

**Panel F contents — six sub-sections:**

**F1 — Bandwidth Allocation Bar**
A horizontal stacked bar showing how total bandwidth is split between the three streams:
```
BANDWIDTH  342 KB/s VIDEO  ████████████████░░░░  70 KB/s LIDAR  ░░░  5 KB/s META
```
Segments: VIDEO in green, LIDAR in amber, META in blue. Labels show KB/s per stream.
Total bandwidth shown as `TOTAL: 417 KB/s`.

**F2 — Three-Tier Degradation Chart (Canvas)**
A live line chart (Canvas 2D, last 60 seconds) showing all three stream rates on a shared time axis:
- Y-axis left: FPS/Hz (0–30)
- Three lines: Video FPS (green), LiDAR Hz (amber, scaled ×3 for visual), Metadata FPS (blue, scaled ×0.2)
- Horizontal dashed threshold line at 15 FPS in red
- X-axis: last 60 seconds (scrolling)
- Legend: `─ VIDEO  ─ LIDAR×3  ─ META÷5`
- This directly visualises the three-tier hierarchy that is the project's main contribution

**F3 — End-to-End Latency Gauge**
Large numeric display: `45 ms` with a colour arc:
- Green arc 0–250ms (5G target)
- Amber arc 250–1000ms
- Red arc 1000–2500ms (satellite NTN range)
- Label: `E2E LATENCY`
- Show P95 estimate below (track the top 5% of values in last 60s): `P95: 78ms`

**F4 — Quality Score**
Large number 0–100 with ring gauge:
```
   94
 QUALITY
 SCORE
```
Colour: green 80–100, amber 60–80, red < 60.
Ring fill proportional to score value.
Show score components below: `FPS×0.4 | LAT×0.3 | DEL×0.3`

**F5 — Inferred Network Profile Badge**
```
INFERRED LINK
[ Good 5G     ]
  87% match
```
Styled as a terminal badge with the profile name inside brackets.
Colour-code by severity: green (Baseline/Good5G), amber (Congested/Poor5G), red (NTN profiles).

**F6 — Frame Loss Indicator**
```
FRAME LOSS
  1.2%
 ▓▓▓░░░░░░░ 
```
Block-bar from 0–10%. Green < 1%, amber 1–5%, red > 5%.

### 3.3 Network Profile Simulator Controls

Add to the Controls panel (Panel C) a new section below the existing toggles:

```
━━━━━━━━━━━━━━━━━━━━━
NETWORK SIMULATION
  ACTIVE: [NONE / BASELINE]

  [BASELINE  ] [GOOD 5G   ]
  [CONG. 5G  ] [POOR 5G   ]
  [SAT. NTN  ] [DEG. NTN  ]

  [■ STOP SIMULATION]
━━━━━━━━━━━━━━━━━━━━━
```

**Backend implementation:**

Add new control endpoints:
```
POST /control/network/start/{profile_name}
POST /control/network/stop
GET  /control/network/status  → {"active": bool, "profile": str|null, "pid": int|null}
```

Profile names: `baseline`, `good_5g`, `congested_5g`, `poor_5g`, `satellite_ntn`, `degraded_ntn`

Map each to the parameters from `network_test.py`. Start the proxy with a single profile and no time limit (run until stopped). Read `network_test.py` to understand how to invoke it programmatically — likely `subprocess` with `--profiles` and `--duration 9999` (effectively unlimited).

If `network_test.py` does not support indefinite run, invoke it with `--duration 3600` (1 hour) as a reasonable upper bound.

Include in the telemetry push:
```json
"active_network_profile": "Good 5G" | null
```

The dashboard shows this prominently next to the inferred profile — comparing what is *configured* vs what is *measured* is useful for verifying the simulation is working.

---

## 4. ENHANCEMENT — MuJoCo Simulator Feed (Priority: Medium)

**Goal:** Capture the live MuJoCo viewer window (which shows a 3D view of the Go2) and stream it into the dashboard as a second video source.

### 4.1 Backend — Simulator Screen Capture

**Strategy:** Use `ffmpeg -f x11grab` to capture the MuJoCo viewer window by finding it via `xdotool`, then serve the output as MJPEG on port 8093.

**New file: `~/Desktop/sim_capture.py`** (separate process, not embedded in dashboard_server.py):

```python
#!/usr/bin/env python3
"""
Captures the MuJoCo viewer window and serves it as MJPEG on port 8093.
Run: python3 sim_capture.py [--port 8093] [--fps 10] [--display :0]
"""
```

Implementation logic:
```python
import subprocess, shlex, time, asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

# 1. Find MuJoCo window geometry using xdotool:
#    xdotool search --name "MuJoCo" getwindowgeometry --shell %1
# Output: X=..., Y=..., WIDTH=..., HEIGHT=...
# Parse these values.
# If xdotool is not installed: apt-get install -y xdotool

# 2. Start ffmpeg to grab that region:
#    ffmpeg -f x11grab -r {fps} -s {W}x{H} -i {DISPLAY}+{X},{Y}
#            -vcodec mjpeg -q:v 5 -f mpjpeg -
# Pipe stdout.

# 3. Serve the raw ffmpeg stdout (which IS a valid MJPEG stream) at GET /sim-stream

# 4. Periodically (every 5s) recheck the window position in case it moved.
#    Update ffmpeg's grab region if window moved significantly (>50px).

# 5. If window not found: return a "SIMULATOR NOT RUNNING" JPEG placeholder.
```

**Key implementation details:**
- ffmpeg command: `ffmpeg -f x11grab -r 10 -s {W}x{H} -i :0.0+{X},{Y} -vcodec mjpeg -q:v 4 -f mpjpeg pipe:1`
- Use `-q:v 4` (lower = higher quality; 4 is good balance)
- Frame rate: 10 fps is sufficient for a physics sim view (no need for 30fps)
- Window search: `subprocess.check_output(['xdotool', 'search', '--name', 'MuJoCo'])` → get WID → `xdotool getwindowgeometry {WID}`
- If xdotool returns multiple windows, take the first
- If ffmpeg is not installed: `print("[SIM] Install ffmpeg: sudo apt-get install -y ffmpeg")` and exit gracefully

**Add `sim_capture.py` startup to `run.sh`:**
```bash
--sim-capture)
    echo "[RUN] Starting MuJoCo screen capture on :8093..."
    python3 ~/Desktop/sim_capture.py > /tmp/sim_capture.log 2>&1 &
    echo $! > /tmp/sim_capture.pid
    echo "[RUN] Sim feed: http://$(hostname -I | awk '{print $1}'):8093/sim-stream"
    ;;
```

Also update `stop.sh` to kill sim_capture if running.

### 4.2 Dashboard — Simulator View Panel

Add a new **Panel G: Simulator View** to the dashboard. This is an optional panel — hidden by default, shown when the simulator capture is available.

**Display:** Small panel in the top area, similar size to the LiDAR panel. Contains:
- `<img id="simFeed" src="http://<host>:8093/sim-stream">` using the same host detection logic as the main video (read from `window.location.hostname`)
- Panel header: `[ ◈ SIM VIEW — FENRIR-1 ]`
- If `8093` is unreachable: show `[ SIMULATOR OFFLINE ]` in dim green, no amber (it's not an error, just not running)
- NO SIGNAL indicator should be dim (not alarming) since sim is optional

**Toggle:** Add a `[▶ SHOW SIM]` / `[■ HIDE SIM]` button in the controls panel. The sim panel is hidden by default and shown on demand (CSS display toggle).

**Note:** The sim feed is served from port 8093 on the Jetson. Over Tailscale, the operator should replace `localhost` with the Tailscale IP — same pattern as the existing video feed host detection logic.

---

## 5. ENHANCEMENT — Robot Control Panel (Priority: Medium)

**Goal:** Add Go2 robot control commands to the dashboard. At this stage: stand and crouch (mapped to real SDK2 calls via a new thin API), plus placeholder buttons for locomotion commands that are not yet implemented.

### 5.1 Pre-Implementation Investigation

**Claude Code must read these files before implementing anything:**
```bash
# Find the actual stand/crouch implementation
cat ~/unitree_mujoco/go2_bridge/*.py 2>/dev/null
find ~/.openclaw -name "*.py" -exec grep -l "stand\|crouch\|SportClient\|BalanceStand" {} \;
# Read whatever files match ^
ls ~/unitree_sdk2_python/example/
```

The implementation must be based on what actually exists in the codebase, not guessed. The most likely patterns are either:
- `SportClient` API: `client.BalanceStand()`, `client.StandDown()`
- Low-level motion commands via DDS publisher

### 5.2 New File: `~/Desktop/go2_api.py`

A standalone FastAPI REST server on port 8094 that wraps Go2 SDK2 commands.

```python
#!/usr/bin/env python3
"""
Go2 control REST API for dashboard integration.
Wraps Unitree SDK2 Python calls.
Run: python3 go2_api.py [--port 8094] [--target sim|robot]
  --target sim:   DOMAIN_ID=1 (MuJoCo simulator)
  --target robot: DOMAIN_ID=0 (real Go2 hardware)
Default: sim
"""
```

**Endpoints:**
```
GET  /status          → {"target": "sim"|"robot", "sdk_connected": bool, "last_command": str}
POST /command/stand   → executes BalanceStand (or equivalent)
POST /command/crouch  → executes StandDown (or equivalent)
POST /command/forward → 501 Not Implemented (placeholder)
POST /command/back    → 501 Not Implemented (placeholder)
POST /command/left    → 501 Not Implemented (placeholder)
POST /command/right   → 501 Not Implemented (placeholder)
POST /command/turn_left  → 501 Not Implemented (placeholder)
POST /command/turn_right → 501 Not Implemented (placeholder)
```

**All endpoints return:**
```json
{"success": true, "command": "stand", "message": "BalanceStand sent to simulator"}
```
or for placeholders:
```json
{"success": false, "command": "forward", "message": "Not implemented — requires onboard gait controller"}
```

**SDK2 initialisation pattern** (fill in from the actual SDK2 code you find):
```python
import sys
sys.path.insert(0, os.path.expanduser('~/unitree_sdk2_python'))
from unitree_sdk2py.core.channel import ChannelFactory
# ChannelFactory.Instance().Init(DOMAIN_ID)
# ... (use what you find in the go2_bridge examples)
```

**Error handling:** If SDK2 import fails or DDS init fails, all command endpoints return `{"success": false, "message": "SDK2 not available: <error>"}` — do not crash.

**Add to `run.sh`:**
```bash
--go2-api)
    TARGET=${2:-sim}
    echo "[RUN] Starting Go2 control API on :8094 (target: $TARGET)..."
    python3 ~/Desktop/go2_api.py --target $TARGET > /tmp/go2_api.log 2>&1 &
    echo $! > /tmp/go2_api.pid
    ;;
```

### 5.3 Dashboard — Robot Control Panel

Extend the existing Controls panel (Panel C) with a new **Robot Control** section:

```
━━━━━━━━━━━━━━━━━━━━━
ROBOT CONTROL
  TARGET: [ SIM | ROBOT ]

  PRIMARY ACTIONS:
  [▲ STAND UP  ] [▼ CROUCH  ]

  LOCOMOTION (UNIMPL.):
  [↑ FWD] [↓ BACK]
  [← LEFT] [→ RIGHT]
  [↺ TURN L] [↻ TURN R]
━━━━━━━━━━━━━━━━━━━━━
```

**Styling:**
- STAND UP and CROUCH: full green border, active appearance
- Locomotion buttons: dim amber border, `opacity: 0.5`, `cursor: not-allowed` — visually disabled but clickable
- On click of locomotion button: show a small toast notification: `[ NOT IMPLEMENTED — Requires onboard gait controller ]` in amber, fades after 2s
- Target toggle: clicking `SIM` or `ROBOT` sends `POST /go2-target/{sim|robot}` to dashboard_server.py which relays to go2_api.py (or restarts it with a different `--target`)

**Backend routing in dashboard_server.py:** Add proxy routes:
```
POST /go2/{command}  → proxy to http://127.0.0.1:8094/command/{command}
GET  /go2/status     → proxy to http://127.0.0.1:8094/status
POST /go2-target/{target} → restart go2_api.py subprocess with new --target flag
```

Use raw asyncio HTTP (same pattern as video proxy, but for short POST requests — use a simple `asyncio.open_connection` to send and receive the JSON response, or since go2_api.py will also be FastAPI, just use a TCP connection).

**Command response handling in frontend:**
- Success: flash the button green for 500ms, show `✓ CMD SENT` in status bar
- Failure/placeholder: show toast with the `message` from response
- Connection failure (go2_api.py not running): show `[ GO2 API OFFLINE ]` in the robot control section

---

## 6. ENHANCEMENT — Dashboard Layout Restructure (Priority: Low)

With the new panels (F: Network, G: Simulator), the existing layout needs adjustment. Suggested updated grid:

```
┌─── STATUS BAR (full width) ────────────────────────────────────────────────┐
├──────────────────────┬─────────────────┬──────────────────┬────────────────┤
│    CAMERA FEED       │  LIDAR POLAR    │  SIM VIEW (G)    │   CONTROLS (C) │
│    (Panel A)         │  (Panel B)      │  [toggleable]    │   + Go2 ctrl   │
│    ~45% width        │  ~20% width     │  ~15% width      │   ~20% width   │
│    ~55% height       │  ~55% height    │  ~55% height     │   ~55% height  │
├────────────┬─────────┴─────────────────┴──────────────────┴────────────────┤
│ DETECTIONS │              NETWORK ANALYTICS (Panel F)                       │
│ (Panel D)  │  BW bar | 3-tier chart | Latency | QScore | Profile | Loss    │
│ ~20% width │  ~80% width                                                    │
│ ~45% height│  ~45% height                                                   │
└────────────┴────────────────────────────────────────────────────────────────┘
```

The Sim View panel (G) is hidden by default (`display: none`) and toggled from the controls. When hidden, the camera feed expands to fill its space (CSS grid auto-sizing).

---

## 7. Acceptance Criteria

### AC-01: Video latency
- Subjective lag on Tailscale < 500ms (noticeably faster than before the httpx fix)

### AC-02: Offline banner
- When all 3 stream `last_seen_s > 10`, amber offline banner appears
- Banner disappears within 2s of any stream recovering

### AC-03: Network bandwidth display
- `video_bw_kbps` within ±20% of actual (calculated from jpeg_bytes × fps)
- Three-tier chart shows distinct separation between video/lidar/metadata lines
- Updates in real time without page refresh

### AC-04: Quality score
- Matches formula: 0.4×(fps/30) + 0.3×(max(0,100-latency/10)) + 0.3×delivery
- Displays 0–100, colour-coded correctly

### AC-05: Profile inference
- Under normal operation (no sim), shows "Baseline" with high confidence
- Under an active network simulation, inferred profile matches the configured profile (verifiable by comparing configured vs measured)

### AC-06: Network profile simulation controls
- Clicking a profile button starts `network_test.py` with that profile
- Active profile shown in controls panel and telemetry push
- `[■ STOP SIMULATION]` halts it cleanly

### AC-07: MuJoCo capture
- When MuJoCo viewer is open on screen AND sim_capture.py is running, the Sim View panel shows a live (10fps) view
- When MuJoCo is not open, panel shows "SIMULATOR OFFLINE" in dim green (not an error state)

### AC-08: Stand/Crouch commands
- `[▲ STAND UP]` POST reaches go2_api.py and executes the SDK2 stand command to the simulator
- Button flashes green on success
- SDK2 command errors shown as toast

### AC-09: Locomotion placeholders
- Clicking forward/back/left/right/turn buttons shows toast: "Not implemented — Requires onboard gait controller"
- Buttons visually distinct from implemented commands (dim, cursor-not-allowed)

### AC-10: Go2 target toggle
- Clicking `[SIM]` vs `[ROBOT]` switches DOMAIN_ID in go2_api.py
- Current target reflected in UI badge

---

## 8. Non-Goals

- No ROS 2 anywhere
- No modification to `stream_server.py`, `lidar_stream.py`, or `inference.py`
- No camera rendering inside MuJoCo (the screen capture approach is sufficient)
- No autonomous locomotion or gait control (placeholder buttons only)
- No WebRTC (MJPEG is sufficient for sim capture too)
- No authentication
- Do not change the mock mode behaviour (it should still work without any hardware)

---

## 9. Implementation Order

1. Read all files per Section 0 inspection tasks — document findings in comments
2. Fix video proxy (Section 1) — test with Tailscale before proceeding
3. Add offline banner (Section 2) — test by stopping all streams
4. Add `network` computed fields to backend telemetry (Section 3.1)
5. Add Network Analytics panel frontend (Section 3.2) — test with mock mode
6. Add network simulation controls (Section 3.3) — requires reading network_test.py
7. Write `sim_capture.py` (Section 4.1) — test: open MuJoCo, run sim_capture, curl localhost:8093/sim-stream
8. Add Sim View panel to dashboard (Section 4.2)
9. Write `go2_api.py` (Section 5.2) — based on actual SDK2 code found in inspection
10. Add robot control panel to dashboard (Section 5.3)
11. Add backend proxy routes for go2 commands
12. Layout restructure if needed (Section 6)
13. Update `run.sh` and `stop.sh` for new services
14. Full integration test: run all services, verify all panels

---

## 10. Mock Mode Updates

Update `--mock` flag to also generate synthetic network analytics:
- `video_bw_kbps`: oscillate between 20–1200 KB/s (simulates profile transitions)
- `e2e_latency_ms`: oscillate between 1–2000ms on a 60-second cycle
- `frame_loss_pct`: oscillate between 0–8%
- `quality_score`: computed from above values
- `inferred_profile`: auto-computed from simulated values
- `active_network_profile`: null (no sim running in mock mode)
- `system_state`: always "online" in mock mode

Go2 command endpoints in mock mode: return `{"success": true, "command": cmd, "message": "Mock: command simulated"}`.

---

## 11. Key Architectural Notes

### Network proxy separation
The userspace proxy (network_test.py) runs on ports 9090/9091/9092 only during experiments. During `--stream` mode, data flows directly on 8090/8091/8092. The dashboard always connects to 8091/8092 directly (no proxy). When a network simulation is active (triggered from the dashboard), the proxy is added as an intermediary layer — the dashboard_server.py must connect to the proxy ports when simulation is active, and to the direct ports when not.

Simplification: rather than switching connection targets, have dashboard_server.py always connect to 8091/8092, and have the network simulator apply constraints at the network level (which it already does via the proxy). The dashboard observes the effects (degraded FPS, higher latency) naturally.

### OpenClaw vs direct SDK2
Do not attempt to route commands through the OpenClaw/Telegram interface. Call the SDK2 Python bindings directly in `go2_api.py`. The inspection in Section 0 will reveal the exact call pattern.

### Tailscale and cross-origin
Over Tailscale, the browser accesses `http://100.90.8.92:8080`. The sim video (8093) and go2_api (8094) are same-host — the dashboard should use `window.location.hostname` as the host for all secondary service connections, not hardcoded IPs.

### Window detection robustness (sim capture)
The MuJoCo window title may vary. Try searching for: "MuJoCo", "unitree", "Go2", "Fenrir". If no window found, fall back to full-screen capture of `:0.0` at a fixed resolution (1280×720) — the user can position the MuJoCo viewer to be visible.
