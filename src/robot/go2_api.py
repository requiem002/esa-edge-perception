#!/usr/bin/env python3
"""
Go2 control REST API for dashboard integration.
Wraps Unitree SDK2 Python bindings (go2_cmd.py patterns).

Run: python3 go2_api.py [--port 8094] [--target sim|robot] [--interface eth0]
  --target sim:   DOMAIN_ID=1, interface lo (MuJoCo simulator)
  --target robot: DOMAIN_ID=0, interface specified by --interface (real Go2)
"""

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ─── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Go2 control REST API")
parser.add_argument("--port", type=int, default=8094)
parser.add_argument("--target", choices=["sim", "robot"], default="sim")
parser.add_argument("--interface", default="eth0", help="Network interface for real robot")
parser.add_argument("--mock", action="store_true", help="Mock mode — no SDK2 required")
args = parser.parse_args()

# ─── State ───────────────────────────────────────────────────────────────────

sdk_available = False
sdk_error = ""
last_command = "none"
last_command_time: float = 0.0
DEBOUNCE_S = 2.0          # reject duplicate commands within this window
GO2_CMD = str(Path.home() / "unitree_mujoco" / "go2_bridge" / "go2_cmd.py")
GO2_WALK = str(Path.home() / "unitree_mujoco" / "go2_bridge" / "go2_mpc_walk.py")

# Walking daemon (go2_mpc_walk.py) — starts on first walk command, listens on UDP.
WALK_UDP_HOST = "127.0.0.1"
WALK_UDP_PORT = 7767
_walk_proc = None            # subprocess.Popen handle for the daemon

# Velocity presets per directional command
WALK_VELOCITIES = {
    "forward":    (0.30, 0.00,  0.00),
    "back":       (-0.20, 0.00, 0.00),
    "left":       (0.00, 0.20,  0.00),
    "right":      (0.00, -0.20, 0.00),
    "turn_left":  (0.00, 0.00,  0.50),
    "turn_right": (0.00, 0.00, -0.50),
}

_cmd_lock = asyncio.Lock()   # prevents concurrent subprocess launches

app = FastAPI()

# ─── SDK check ───────────────────────────────────────────────────────────────

def check_sdk():
    global sdk_available, sdk_error
    try:
        sdk_path = str(Path.home() / "unitree_sdk2_python")
        sys.path.insert(0, sdk_path)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        sdk_available = True
        print(f"[go2_api] SDK2 available at {sdk_path}")
    except Exception as e:
        sdk_available = False
        sdk_error = str(e)
        print(f"[go2_api] SDK2 not available: {e}")

if not args.mock:
    check_sdk()
else:
    sdk_available = True
    print("[go2_api] Mock mode — no SDK2 required")

# ─── Command execution ────────────────────────────────────────────────────────

async def run_go2_cmd(action: str) -> dict:
    """Execute go2_cmd.py as a subprocess (fire-and-forget) with debounce."""
    global last_command, last_command_time

    if not Path(GO2_CMD).exists():
        return {"success": False, "command": action,
                "message": f"go2_cmd.py not found at {GO2_CMD}"}

    # Debounce: each pose transition takes ~1.5s — reject rapid repeats
    now = time.time()
    elapsed = now - last_command_time
    if elapsed < DEBOUNCE_S:
        wait = round(DEBOUNCE_S - elapsed, 1)
        return {"success": False, "command": action,
                "message": f"Motion in progress — retry in {wait}s"}

    # Lock prevents two commands racing through the debounce check simultaneously
    if _cmd_lock.locked():
        return {"success": False, "command": action,
                "message": "Another command is already being dispatched"}

    async with _cmd_lock:
        try:
            subprocess.Popen(
                [sys.executable, GO2_CMD, action],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            last_command = action
            last_command_time = time.time()
            target_label = "simulator" if args.target == "sim" else "real robot"
            return {"success": True, "command": action,
                    "message": f"{action.capitalize()} sent to {target_label}"}
        except Exception as e:
            return {"success": False, "command": action, "message": str(e)}


# ─── Walk daemon management ──────────────────────────────────────────────────

def _walk_alive() -> bool:
    return _walk_proc is not None and _walk_proc.poll() is None


def _walk_send(line: str, timeout: float = 1.0) -> str:
    """One-shot UDP request to the walk daemon. Returns reply text or ''."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(line.encode(), (WALK_UDP_HOST, WALK_UDP_PORT))
        data, _ = s.recvfrom(512)
        return data.decode("utf-8", errors="ignore").strip()
    except (socket.timeout, OSError):
        return ""
    finally:
        s.close()


def _walk_ensure_running() -> bool:
    """Start go2_walk.py daemon if it isn't already running. Returns True
    once the UDP listener responds to a status probe.

    Cold-start path: if the daemon isn't already up, the robot is likely
    sitting in the MJCF default keyframe (deeply crouched, splayed legs).
    Running go2_cmd.py stand first brings it up cleanly with a smooth
    interpolation; then the daemon takes over and holds the stand pose.
    Without this, the daemon's own startup ramp can fail to lift the
    robot off the floor and it ends up tipped over.
    """
    global _walk_proc
    if _walk_alive() and _walk_send("status"):
        return True
    if not Path(GO2_WALK).exists():
        return False
    if Path(GO2_CMD).exists():
        try:
            subprocess.run(
                [sys.executable, GO2_CMD, "stand"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8.0,
            )
        except subprocess.TimeoutExpired:
            pass
    _walk_proc = subprocess.Popen(
        [sys.executable, GO2_WALK],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Detach so the daemon survives if go2_api.py exits cleanly,
        # but the same process group lets us kill it on shutdown.
        start_new_session=True,
    )
    # Wait up to 15s — MPC daemon needs ~3s to init solver on Jetson
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if _walk_send("status", timeout=0.5):
            return True
        time.sleep(0.2)
    return False


def _walk_kill():
    """Stop the daemon (graceful UDP quit, then terminate as fallback)."""
    global _walk_proc
    if _walk_alive():
        _walk_send("quit", timeout=0.3)
        try:
            _walk_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _walk_proc.terminate()
    _walk_proc = None


async def run_walk_cmd(direction: str, vx: float, vy: float, vyaw: float) -> dict:
    """Start the walk daemon if needed and command it to walk."""
    global last_command, last_command_time
    if not _walk_ensure_running():
        return {"success": False, "command": direction,
                "message": "Failed to start go2_walk daemon"}
    reply = _walk_send(f"walk {vx} {vy} {vyaw}")
    if not reply.startswith("ok"):
        return {"success": False, "command": direction,
                "message": f"daemon rejected command: {reply or '(no reply)'}"}
    last_command = direction
    last_command_time = time.time()
    return {"success": True, "command": direction,
            "message": f"Walking {direction} (vx={vx} vy={vy} vyaw={vyaw})"}


async def run_walk_stop() -> dict:
    """Stop walking — daemon transitions back to STAND_POSE and holds."""
    global last_command, last_command_time
    if not _walk_alive():
        return {"success": True, "command": "stop",
                "message": "Daemon not running — nothing to stop"}
    reply = _walk_send("stop")
    if not reply.startswith("ok"):
        return {"success": False, "command": "stop",
                "message": f"daemon rejected stop: {reply or '(no reply)'}"}
    last_command = "stop"
    last_command_time = time.time()
    return {"success": True, "command": "stop",
            "message": "Stopped — robot holding stand pose"}


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    elapsed = time.time() - last_command_time
    walk_state = ""
    if _walk_alive():
        walk_state = _walk_send("status", timeout=0.3)
    return {
        "target": args.target,
        "sdk_connected": sdk_available,
        "sdk_error": sdk_error if not sdk_available else "",
        "last_command": last_command,
        "ready": elapsed >= DEBOUNCE_S,
        "go2_cmd_path": GO2_CMD,
        "go2_walk_path": GO2_WALK,
        "walk_daemon_alive": _walk_alive(),
        "walk_daemon_state": walk_state,
    }

@app.post("/command/stand")
async def cmd_stand():
    global last_command, last_command_time
    if args.mock:
        last_command = "stand"
        return {"success": True, "command": "stand", "message": "Mock: stand simulated"}
    if not sdk_available:
        return JSONResponse(
            {"success": False, "command": "stand", "message": f"SDK2 not available: {sdk_error}"},
            status_code=503)
    # Always route stand through the daemon. The daemon's STAND state
    # continuously publishes STAND_POSE at 100 Hz; the one-shot go2_cmd.py
    # only holds the pose for ~2s and then the robot drifts under gravity
    # because the sim's LowCmdHandler needs continuous messages.
    if not _walk_ensure_running():
        return JSONResponse(
            {"success": False, "command": "stand",
             "message": "Failed to start go2_walk daemon"},
            status_code=500)
    reply = _walk_send("stand")
    if not reply.startswith("ok"):
        return {"success": False, "command": "stand",
                "message": f"daemon rejected stand: {reply or '(no reply)'}"}
    last_command = "stand"
    last_command_time = time.time()
    return {"success": True, "command": "stand",
            "message": "Standing — daemon holding pose"}

@app.post("/command/crouch")
async def cmd_crouch():
    global last_command
    if args.mock:
        last_command = "crouch"
        return {"success": True, "command": "crouch", "message": "Mock: crouch simulated"}
    if not sdk_available:
        return JSONResponse(
            {"success": False, "command": "crouch", "message": f"SDK2 not available: {sdk_error}"},
            status_code=503)
    # Crouch isn't supported by the walk daemon; kill it first so the
    # one-shot go2_cmd.py crouch isn't fighting another writer on rt/lowcmd.
    if _walk_alive():
        _walk_kill()
    return await run_go2_cmd("crouch")


def _walk_endpoint(direction: str):
    """Build a FastAPI endpoint that translates a directional command
    into a velocity tuple and dispatches to the walk daemon."""
    vx, vy, vyaw = WALK_VELOCITIES[direction]

    async def _endpoint():
        if args.mock:
            return {"success": True, "command": direction,
                    "message": f"Mock: walk {direction} simulated"}
        if not sdk_available:
            return JSONResponse(
                {"success": False, "command": direction,
                 "message": f"SDK2 not available: {sdk_error}"},
                status_code=503)
        return await run_walk_cmd(direction, vx, vy, vyaw)

    _endpoint.__name__ = f"cmd_{direction}"
    return _endpoint


for _dir in WALK_VELOCITIES:
    app.post(f"/command/{_dir}")(_walk_endpoint(_dir))


@app.post("/command/stop")
async def cmd_stop():
    if args.mock:
        return {"success": True, "command": "stop", "message": "Mock: stop simulated"}
    if not sdk_available:
        return JSONResponse(
            {"success": False, "command": "stop", "message": f"SDK2 not available: {sdk_error}"},
            status_code=503)
    return await run_walk_stop()


@app.post("/command/{command}")
async def cmd_other(command: str):
    return JSONResponse(
        {"success": False, "command": command,
         "message": f"Unknown command '{command}'. Available: stand, crouch, "
                    "stop, " + ", ".join(WALK_VELOCITIES)},
        status_code=404)

# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print(f"[go2_api] Serving on port {args.port} (target={args.target})")


@app.on_event("shutdown")
async def shutdown():
    # Don't leave the walk daemon running and writing to rt/lowcmd after
    # the API exits — the dashboard would have no way to stop it.
    if _walk_alive():
        print("[go2_api] Stopping walk daemon...")
        _walk_kill()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
