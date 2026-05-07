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
GO2_CMD = str(Path.home() / "unitree_mujoco" / "go2_bridge" / "go2_cmd.py")

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
    """Execute go2_cmd.py as a subprocess (fire-and-forget)."""
    global last_command
    if not Path(GO2_CMD).exists():
        return {"success": False, "command": action,
                "message": f"go2_cmd.py not found at {GO2_CMD}"}
    try:
        # Run in background — motion takes ~1.5s, don't block the API
        subprocess.Popen(
            [sys.executable, GO2_CMD, action],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        last_command = action
        target_label = "simulator" if args.target == "sim" else "real robot"
        return {"success": True, "command": action,
                "message": f"{action.capitalize()} sent to {target_label}"}
    except Exception as e:
        return {"success": False, "command": action, "message": str(e)}

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "target": args.target,
        "sdk_connected": sdk_available,
        "sdk_error": sdk_error if not sdk_available else "",
        "last_command": last_command,
        "go2_cmd_path": GO2_CMD,
    }

@app.post("/command/stand")
async def cmd_stand():
    global last_command
    if args.mock:
        last_command = "stand"
        return {"success": True, "command": "stand", "message": "Mock: stand simulated"}
    if not sdk_available:
        return JSONResponse(
            {"success": False, "command": "stand", "message": f"SDK2 not available: {sdk_error}"},
            status_code=503)
    return await run_go2_cmd("stand")

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
    return await run_go2_cmd("crouch")

@app.post("/command/{command}")
async def cmd_other(command: str):
    return JSONResponse(
        {"success": False, "command": command,
         "message": "Not implemented — requires onboard gait controller"},
        status_code=501)

# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print(f"[go2_api] Serving on port {args.port} (target={args.target})")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
