#!/usr/bin/env python3
"""
Quick status dashboard for the YOLO inference deployment.

Run on the host (not inside the container):
    python3 ~/Desktop/status.py
"""

import shutil
import subprocess
from pathlib import Path


def run(cmd):
    """Run a shell command, return (stdout, return_code)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    return r.stdout.strip(), r.returncode


def read_thermal(zone_type):
    """Read a thermal zone by type name (e.g. 'gpu-thermal')."""
    for zone in sorted(Path("/sys/devices/virtual/thermal/").glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == zone_type:
                return int((zone / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError, TypeError):
            continue
    return None


def main():
    print(f"{'=' * 55}")
    print("  YOLO INFERENCE STATUS")
    print(f"{'=' * 55}")

    # -- Container --
    out, rc = run("docker inspect -f '{{.State.Status}}' yolo-saad 2>/dev/null")
    container_up = (out == "running")
    print(f"\n  Container (yolo-saad) : {'RUNNING' if container_up else 'STOPPED'}")

    # -- Inference process --
    if container_up:
        _, rc = run("docker exec yolo-saad pgrep -f 'python3 /workspace/inference.py'")
        inference_up = (rc == 0)
    else:
        inference_up = False
    print(f"  Inference process    : {'RUNNING' if inference_up else 'NOT RUNNING'}")

    # -- Temperatures --
    gpu_t = read_thermal("gpu-thermal")
    cpu_t = read_thermal("cpu-thermal")
    tj_t = read_thermal("tj-thermal")
    print(f"\n  GPU Temperature      : {gpu_t:.1f} C" if gpu_t else
          "  GPU Temperature      : N/A")
    print(f"  CPU Temperature      : {cpu_t:.1f} C" if cpu_t else
          "  CPU Temperature      : N/A")
    print(f"  Tj  Temperature      : {tj_t:.1f} C" if tj_t else
          "  Tj  Temperature      : N/A")

    # -- Memory --
    out, _ = run("free -m | awk '/Mem:/ {printf \"%d/%d MB (%.0f%%)\", $3, $2, $3/$2*100}'")
    print(f"\n  System Memory        : {out}")

    if container_up:
        gpu_mem, _ = run(
            "docker exec yolo-saad bash -c "
            "\"python3 -c \\\"import torch; a=torch.cuda.mem_get_info(); "
            "print(f'{(a[1]-a[0])//1048576}/{a[1]//1048576} MB')\\\" 2>/dev/null\""
        )
        if gpu_mem:
            print(f"  GPU Memory (used/tot): {gpu_mem}")

    # -- Disk --
    usage = shutil.disk_usage("/home/esa_jetson/Desktop")
    used_pct = usage.used / usage.total * 100
    free_mb = usage.free / (1024 ** 2)
    warn = " *** LOW ***" if free_mb < 500 else ""
    print(f"\n  Disk (/)             : {used_pct:.0f}% used, {free_mb:.0f} MB free{warn}")

    # -- Latest detection --
    output_dir = Path("/home/esa_jetson/Desktop/output")
    csv_files = sorted(output_dir.glob("detections_*.csv"),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True) if output_dir.exists() else []
    if csv_files:
        latest = csv_files[0]
        print(f"\n  Latest CSV           : {latest.name}")
        try:
            lines = latest.read_text().strip().split("\n")
            if len(lines) > 1:
                parts = lines[-1].split(",")
                print(f"  Last detection       : {parts[2]} (conf {parts[3]}) at {parts[0]}")
            else:
                print(f"  Last detection       : (no detections yet)")
        except (IndexError, OSError):
            print(f"  Last detection       : (read error)")
    else:
        print(f"\n  Last detection       : No output files found")

    # -- Output file counts --
    if output_dir.exists():
        vids = list(output_dir.glob("detect_*.avi"))
        csvs = list(output_dir.glob("detections_*.csv"))
        print(f"  Output files         : {len(vids)} videos, {len(csvs)} CSVs")

    # -- Systemd service --
    svc_out, svc_rc = run("systemctl is-active yolo-inference.service 2>/dev/null")
    if svc_rc == 0:
        status_label = svc_out.upper()
    else:
        status_label = "NOT INSTALLED" if "could not be found" in svc_out or not svc_out else svc_out.upper()
    print(f"\n  Systemd service      : {status_label}")

    print(f"\n{'=' * 55}")


if __name__ == "__main__":
    main()
