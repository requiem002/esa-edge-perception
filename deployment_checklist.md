# Unitree Go2 Deployment Checklist

What to do when the Go2 arrives and you plug the Jetson into it.

The pipeline currently expects a USB webcam at `/dev/video0`. The Go2's onboard cameras are MIPI/CSI-attached and present a different device interface, so the camera source needs adjusting before anything else.

---

## 0. Pre-arrival sanity check (do this **before** the robot ships)

Run these on the Jetson alone — they verify the perception pipeline still works in isolation:

```bash
~/Desktop/run.sh --status        # everything green?
~/Desktop/run.sh --no-display    # 5 s of inference with USB webcam
~/Desktop/stop.sh                # clean shutdown
```

If any of those fail, fix them now. Don't debug them after the robot is on the desk.

---

## 1. Identify the Go2's camera

Once the robot is powered and the Jetson is wired in:

```bash
# What v4l2 devices showed up?
ls /dev/video*

# What capture formats does each one report?
for d in /dev/video*; do
    echo "=== $d ==="
    v4l2-ctl -d "$d" --list-formats-ext 2>&1 | head -20
done
```

The Go2 has two MIPI/CSI cameras (front fisheye + chin). On Jetson Orin, MIPI cameras typically appear as **`/dev/video0`–`/dev/video3`** through `nvargus-daemon`, not as standard UVC devices, so `v4l2-ctl --list-formats` may report unusual pixel formats (NV12, BG10, etc.). If `cv2.VideoCapture(0)` opens the device but returns black/garbled frames, you almost certainly need a GStreamer pipeline instead of a raw V4L2 read:

```python
# Replace cv2.VideoCapture(0) with:
gst = ("nvarguscamerasrc sensor-id=0 ! "
       "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
       "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
       "video/x-raw,format=BGR ! appsink")
cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
```

`inference.py --source` already accepts a string source, so a GStreamer pipeline can be passed via `--source "<gst pipeline>"` — no code changes needed once the right pipeline is found.

If the Go2 instead exposes a **plain UVC USB webcam** (the Edu kit sometimes does), the existing code path works — just verify with `v4l2-ctl -d /dev/videoN --list-formats` that you see `MJPG` or `YUYV`.

## 2. Update the Docker `--device` flag

The `yolo-saad` container currently maps `/dev/video0`. If the Go2 cameras land on different devices (e.g., `/dev/video1` and `/dev/video2`), the container won't see them.

The container is intentionally not recreated by the launcher (preserves state). On the Go2, **either**:

**Option A — recreate the container** (clean approach, takes ~30 s):

```bash
docker rm -f yolo-saad
docker run -d --name yolo-saad --ipc=host --runtime nvidia --network host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --device /dev/video0 --device /dev/video1 --device /dev/video2 \
    --device /dev/ttyTHS1 \
    -v ~/Desktop:/workspace \
    yolo-orin:v2 sleep infinity
```

**Option B — bind-mount the device into the existing container** at runtime (faster but per-session): add `--device-add /dev/videoN` to the docker exec call. This requires Docker ≥ 20.10 and the running container kernel cap.

Option A is the recommended path — it's 30 seconds and idempotent.

## 3. Network for streaming to remote operator

The streaming server inside `yolo-saad` already binds to `0.0.0.0` on ports 8090 (MJPEG) and 8091 (TCP metadata). LiDAR is on host port 8092. The container uses `--network host`, so these are reachable on whatever IP the Jetson has.

Steps:
1. **Get the Jetson's IP**: `ip -4 addr show | grep inet`
2. **Punch firewall holes**: `sudo ufw allow 8090:8092/tcp` (if `ufw` is on; off by default on JetPack)
3. **Test from the operator side**:
   - `curl -I http://<jetson-ip>:8090/stream` should return `200 OK` with `Content-Type: multipart/x-mixed-replace`
   - `nc -zv <jetson-ip> 8091` and `nc -zv <jetson-ip> 8092` should both succeed

For the field deployment over 5G/NTN, the operator should connect through **the same TCP ports** — the userspace proxy in `network_test.py` is for *measurement*, not deployment. Real-world bandwidth limits are imposed by the carrier.

## 4. Systemd service auto-start on boot

Already installed and `enable`d (state: `enabled; vendor preset: enabled`). On next boot the Jetson will:
1. Wait for `docker.service` and network
2. `docker start yolo-saad`
3. Run `inference.py --no-display` inside it

Verify with:
```bash
sudo systemctl reboot
# wait, log back in
journalctl -u yolo-inference.service -f --since "5 min ago"
```

If the service unit fails after a Jetson update or container rename, edit `/etc/systemd/system/yolo-inference.service` and `sudo systemctl daemon-reload`.

The service does **not** start `stream_server.py` or `lidar_stream.py` — those are operator-controlled. Use `~/Desktop/run.sh --stream` and `~/Desktop/lidar_stream.py` (or wrap them in their own services if always-on streaming is wanted).

## 5. Switching simulation ↔ real robot

On the Jetson, the perception pipeline is the same in both cases — only the camera source changes:

| Mode | `--source` | Notes |
|------|-----------|-------|
| USB webcam | `0` | Default; `/dev/video0` |
| MuJoCo simulator | `2` | Requires `sudo modprobe v4l2loopback devices=1 video_nr=2`, then `python3 ~/unitree_mujoco/sim_camera_feed.py &` (host) |
| Go2 MIPI camera | a GStreamer string (see §1) | Once verified |
| Pre-recorded video | a path | Useful for regression testing |

For the **Unitree SDK side** (control, not perception), edit `~/unitree_mujoco/simulate_python/config.py`:
- `DOMAIN_ID = 1` → simulator
- `DOMAIN_ID = 0` → real robot

That switch is independent of the perception pipeline.

## 6. First tests once the robot is on the bench

In this exact order — each step is cheap and isolates one failure mode:

1. **Power, no movement, no inference**: just turn on the Go2 and verify it doesn't fault. Confirm the Jetson boots and reaches a login.
2. **Cameras enumerate**: `ls /dev/video*` and `v4l2-ctl --list-devices` from §1.
3. **YOLO sees frames from the new camera**: `~/Desktop/run.sh --no-display --source "<new src>"` for 10 s. Look at `~/Desktop/output/detect_*.avi` in VLC.
4. **LiDAR**: `python3 ~/Desktop/lidar_stream.py --log /tmp/l.csv` for 10 s. Need ≥ 80 scans for the 360° rotation (LD19 is 10 Hz). If it complains about `/dev/ttyTHS1`, check `dmesg | grep tty` — the LiDAR may have moved.
5. **End-to-end stream over LAN**: `./run.sh --stream` then on the operator laptop point a browser at `http://<jetson-ip>:8090/stream`. You should see boxes drawn live.
6. **Tiny network experiment**: `python3 ~/Desktop/network_test.py --duration 10 --profiles 1_baseline,5_satellite_ntn` — 30 s to confirm the proxy still works on the new device topology.
7. **Reboot test**: `sudo systemctl reboot`, log back in, `~/Desktop/run.sh --status`. Inference process should be RUNNING (started by systemd).
8. **Detection demo**: walk a person past the front camera, check `~/Desktop/output/detections_*.csv` has rows.

If 1–6 pass, you have the perception layer. 7 covers autonomous boot. 8 covers field readiness.

## 7. Known limitations and workarounds

| Issue | Impact | Workaround |
|-------|--------|------------|
| Tegra kernel lacks `sch_netem` | Can't use `tc netem` for network emulation | Userspace proxy in `network_test.py` is the deliberate replacement; works equivalently for thesis measurements |
| Container at `dustynv/l4t-pytorch:r36.2.0` base | Re-export of `.engine` file required if base image changes | Document the export step (`yolo export model=… format=engine`) on first deploy |
| TensorRT engines are device-specific | `best.engine` from another Jetson won't load | Always re-export on the target Jetson |
| Disk on this Jetson runs at ~93% | Risk of out-of-space mid-experiment | `LOG-3` halts video at 200 MB free; add an external NVMe before extended deployments |
| Go2 MIPI cameras need GStreamer | `cv2.VideoCapture(0)` may return black frames | Use the `nvarguscamerasrc` pipeline in §1 |
| `best.pt` / `best2.pt` (KITTI fine-tunes) are broken | Not safe to deploy | Stay on `yolo26n.pt` until MDL-5 is re-executed with frozen backbone |
| LiDAR ↔ camera fusion not implemented | `INT-7` (person within 2 m) can't trigger | LiDAR data is exposed on TCP :8092; downstream code can subscribe and fuse |
| `stream_server` and `inference.py` both want the camera | Only one can run at a time | `~/Desktop/stop.sh` always cleans up; `run.sh --stream` and `run.sh` are mutually exclusive |
| No replay-on-reconnect for telemetry | `NET-4` partial | Operator-side replay from `detections_*.csv` works as offline buffer |

---

## Quick reference: commands you'll actually type

```bash
# Live inference (with display)
~/Desktop/run.sh

# Headless inference (default for boot service)
~/Desktop/run.sh --no-display

# Streaming server (for operator viewing)
~/Desktop/run.sh --stream

# Full network experiment
~/Desktop/run.sh --experiment

# Status
~/Desktop/run.sh --status

# Stop everything
~/Desktop/stop.sh

# Watch the systemd service
journalctl -u yolo-inference.service -f
```
