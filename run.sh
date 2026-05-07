#!/usr/bin/env bash
# Universal launcher for the YOLO perception pipeline.
#
# Modes:
#   ./run.sh                   inference with live display (default)
#   ./run.sh --no-display      headless inference (saves video + CSV)
#   ./run.sh --stream          start stream_server.py for network experiments
#   ./run.sh --status          print status.py dashboard
#   ./run.sh --experiment      full network degradation experiment
#                              (6 profiles, 60s, 3 repeats)
#   ./run.sh --dashboard       start mission control dashboard on :8080
#   ./run.sh --help            show this message
#
# Any extra args are forwarded to inference.py (default mode only):
#   ./run.sh --conf 0.4 --source /workspace/test.mp4
#   ./run.sh --no-display --thermal-throttle 80
#
# Requires: yolo-saad container (started automatically if stopped),
# camera at /dev/video0, optional LIDAR at /dev/ttyTHS1.

set -euo pipefail

CONTAINER="yolo-saad"
DESKTOP="${HOME}/Desktop"

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

# ── Helpers ──────────────────────────────────────────────────────────

ensure_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
            echo "[run.sh] Starting stopped container '${CONTAINER}'..."
            docker start "${CONTAINER}" > /dev/null
        else
            echo "[run.sh] ERROR: container '${CONTAINER}' does not exist."
            echo "         Create it from yolo-orin:v2 (see CLAUDE.md)."
            exit 1
        fi
    fi
}

allow_x_display() {
    # Best-effort — silent if no X server (e.g. SSH session).
    xhost +local: > /dev/null 2>&1 || true
}

# ── Mode dispatch ────────────────────────────────────────────────────

MODE="inference"
INFERENCE_EXTRA=()

# Peek at first arg to choose mode; pass everything else through.
if [[ ${#@} -gt 0 ]]; then
    case "$1" in
        --help|-h)        usage ;;
        --stream)         MODE="stream"      ; shift ;;
        --status)         MODE="status"      ; shift ;;
        --experiment)     MODE="experiment"  ; shift ;;
        --dashboard)      MODE="dashboard"   ; shift ;;
        --sim-capture)    MODE="sim-capture" ; shift ;;
        --sim-cam)        MODE="sim-cam"     ; shift ;;
        --go2-api)        MODE="go2-api"     ; shift ;;
        --sim)            MODE="sim"         ; shift ;;
        --no-display)     MODE="inference"   ; INFERENCE_EXTRA+=("$1") ; shift ;;
        *)                MODE="inference"  ;;
    esac
fi
INFERENCE_EXTRA+=("$@")

case "${MODE}" in

    # ── Default: live inference inside container ─────────────────────
    inference)
        ensure_container
        allow_x_display
        echo "[run.sh] Running inference.py (display: $(
            [[ " ${INFERENCE_EXTRA[*]} " == *" --no-display "* ]] && echo off || echo on
        ))..."
        echo "         Args: ${INFERENCE_EXTRA[*]:-<none>}"
        echo "         Press Ctrl+C or 'q' (in window) to stop."
        echo
        exec docker exec -it \
            -e DISPLAY="${DISPLAY:-:0}" \
            "${CONTAINER}" \
            python3 /workspace/inference.py "${INFERENCE_EXTRA[@]}"
        ;;

    # ── Streaming server for network experiments ─────────────────────
    stream)
        ensure_container
        OUT_DIR="${DESKTOP}/network_results/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "${OUT_DIR}"
        echo "[run.sh] Starting stream_server.py inside ${CONTAINER}..."
        echo "         MJPEG http://localhost:8090/stream"
        echo "         Metadata TCP localhost:8091"
        echo "         Server log: ${OUT_DIR}/server_log.csv"
        echo "         Press Ctrl+C to stop."
        echo
        exec docker exec -it "${CONTAINER}" \
            python3 /workspace/stream_server.py \
                --log "/workspace/network_results/$(basename "${OUT_DIR}")/server_log.csv" \
                "${INFERENCE_EXTRA[@]}"
        ;;

    # ── Status dashboard ─────────────────────────────────────────────
    status)
        exec python3 "${DESKTOP}/status.py"
        ;;

    # ── Full network degradation experiment ──────────────────────────
    experiment)
        ensure_container
        OUT_DIR="${DESKTOP}/network_results/full_experiment_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "${OUT_DIR}"
        SERVER_LOG="${OUT_DIR}/server_log.csv"
        LIDAR_LOG="${OUT_DIR}/lidar_server_log.csv"

        # Make sure no stream_server is already running inside the container
        docker exec "${CONTAINER}" pkill -f stream_server.py 2>/dev/null || true
        pkill -f lidar_stream.py 2>/dev/null || true
        sleep 1

        echo "[run.sh] Starting stream_server.py in container..."
        docker exec -d "${CONTAINER}" \
            python3 /workspace/stream_server.py \
                --log "/workspace/network_results/$(basename "${OUT_DIR}")/server_log.csv"

        if [[ -e /dev/ttyTHS1 ]]; then
            echo "[run.sh] Starting lidar_stream.py on host..."
            nohup python3 "${DESKTOP}/lidar_stream.py" \
                --log "${LIDAR_LOG}" \
                > "${OUT_DIR}/lidar_stream.log" 2>&1 &
        else
            echo "[run.sh] No LIDAR detected (/dev/ttyTHS1 missing) — camera-only run."
        fi

        # Wait for servers to settle (model load + serial open)
        echo "[run.sh] Waiting 8s for servers to come up..."
        sleep 8

        echo "[run.sh] Running network_test.py..."
        python3 "${DESKTOP}/network_test.py" \
            --duration 60 --repeats 3 \
            --output-dir "${OUT_DIR}"

        echo "[run.sh] Stopping servers..."
        docker exec "${CONTAINER}" pkill -f stream_server.py 2>/dev/null || true
        pkill -f lidar_stream.py 2>/dev/null || true

        echo "[run.sh] Generating plots..."
        python3 "${DESKTOP}/analyse_results.py" \
            --input "${OUT_DIR}/summary.csv"

        echo
        echo "[run.sh] Experiment complete: ${OUT_DIR}"
        ;;

    # ── Mission Control Dashboard ────────────────────────────────────
    dashboard)
        echo "[run.sh] Starting mission control dashboard on :8080..."
        pip3 install -q fastapi uvicorn wsproto 2>/dev/null
        nohup python3 "${DESKTOP}/dashboard_server.py" \
            > "${DESKTOP}/dashboard.log" 2>&1 &
        echo $! > /tmp/dashboard.pid
        sleep 1
        JETSON_IP=$(hostname -I | awk '{print $1}')
        echo "[run.sh] Dashboard: http://${JETSON_IP}:8080"
        ;;

    # ── MuJoCo screen capture ────────────────────────────────────────
    sim-capture)
        export DISPLAY="${DISPLAY:-:1}"
        echo "[run.sh] Starting MuJoCo screen capture on :8093..."
        DISPLAY="${DISPLAY}" nohup python3 "${DESKTOP}/sim_capture.py" \
            > /tmp/sim_capture.log 2>&1 &
        echo $! > /tmp/sim_capture.pid
        sleep 1
        JETSON_IP=$(hostname -I | awk '{print $1}')
        echo "[run.sh] Sim feed: http://${JETSON_IP}:8093/sim-stream"
        ;;

    # ── MuJoCo virtual camera → /dev/video2 (physical stays on /dev/video0) ──
    sim-cam)
        SIM_CAM_SCRIPT="${HOME}/unitree_mujoco/sim_camera_feed.py"
        if [[ ! -f "${SIM_CAM_SCRIPT}" ]]; then
            echo "[run.sh] ERROR: ${SIM_CAM_SCRIPT} not found."
            exit 1
        fi
        echo "[run.sh] Setting up MuJoCo virtual webcam on /dev/video2..."
        echo "         Physical camera (/dev/video0) and LiDAR are unaffected."
        echo
        # Load v4l2loopback if not already present
        if ! lsmod | grep -q v4l2loopback; then
            echo "[run.sh] Loading v4l2loopback kernel module (requires sudo)..."
            sudo modprobe v4l2loopback devices=1 video_nr=2 \
                card_label="MuJoCo_Sim" exclusive_caps=1 || {
                echo "[run.sh] ERROR: Could not load v4l2loopback."
                echo "         Install first: sudo apt install v4l2loopback-dkms v4l2loopback-utils"
                exit 1
            }
            echo "[run.sh] v4l2loopback loaded — /dev/video2 created."
        else
            echo "[run.sh] v4l2loopback already loaded."
        fi
        if ! python3 -c "import pyfakewebcam" 2>/dev/null; then
            echo "[run.sh] Installing pyfakewebcam..."
            pip3 install -q --user pyfakewebcam
        fi
        nohup python3 "${SIM_CAM_SCRIPT}" \
            > /tmp/sim_cam.log 2>&1 &
        echo $! > /tmp/sim_cam.pid
        sleep 2
        echo
        echo "[run.sh] Virtual cam active: /dev/video2 (MuJoCo front camera at 30fps)"
        echo "[run.sh] To run YOLO on the simulated camera, restart the container"
        echo "         with /dev/video2 mapped, then:"
        echo "           ./run.sh --source /dev/video2"
        echo "         or (inside container):"
        echo "           python3 /workspace/stream_server.py --source 2"
        echo
        echo "[run.sh] Physical camera (./run.sh) and LiDAR (./run.sh --stream)"
        echo "         continue to use /dev/video0 as normal."
        ;;

    # ── Go2 control API ──────────────────────────────────────────────
    go2-api)
        TARGET="${INFERENCE_EXTRA[0]:-sim}"
        echo "[run.sh] Starting Go2 control API on :8094 (target: ${TARGET})..."
        nohup python3 "${DESKTOP}/go2_api.py" \
            --target "${TARGET}" \
            > /tmp/go2_api.log 2>&1 &
        echo $! > /tmp/go2_api.pid
        echo "[run.sh] Go2 API: http://127.0.0.1:8094/status"
        ;;

    # ── Simulator integration (MuJoCo + Go2 API + screen capture) ────
    sim)
        export DISPLAY="${DISPLAY:-:0}"
        echo
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  FENRIR-1 SIMULATOR SETUP"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo
        echo "  You need to start the MuJoCo simulator MANUALLY first."
        echo "  Open a separate terminal and run:"
        echo
        echo "    cd ~/unitree_mujoco/simulate_python"
        echo "    python3 unitree_mujoco.py"
        echo
        echo "  The MuJoCo viewer window will open showing the Go2 robot."
        echo "  The simulator uses DOMAIN_ID=1 on loopback (lo interface)"
        echo "  and communicates with go2_api.py via DDS."
        echo
        echo "  (Optional) For YOLO on simulated camera via virtual webcam:"
        echo "    sudo modprobe v4l2loopback devices=1 video_nr=2 card_label=MuJoCo_Sim exclusive_caps=1"
        echo "    python3 ~/unitree_mujoco/sim_camera_feed.py"
        echo "    # then: ./run.sh --stream  (adds --source 2 inside container)"
        echo
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        read -p "  Press Enter once the MuJoCo viewer window is open... "
        echo
        echo "[run.sh] Starting MuJoCo screen capture on :8093..."
        DISPLAY="${DISPLAY:-:1}" nohup python3 "${DESKTOP}/sim_capture.py" \
            > /tmp/sim_capture.log 2>&1 &
        echo $! > /tmp/sim_capture.pid
        sleep 2
        echo "[run.sh] Starting Go2 control API on :8094 (target: sim)..."
        nohup python3 "${DESKTOP}/go2_api.py" \
            --target sim \
            > /tmp/go2_api.log 2>&1 &
        echo $! > /tmp/go2_api.pid
        JETSON_IP=$(hostname -I | awk '{print $1}')
        echo
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Simulator services ready:"
        echo "    Sim view:  http://${JETSON_IP}:8093/sim-stream"
        echo "    Go2 API:   http://${JETSON_IP}:8094/status"
        echo "  Open the dashboard and click [▶ SHOW SIM] to see the viewer."
        echo "  Use [▲ STAND] / [▼ CROUCH] in the Robot Control panel."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo
        ;;

    *)
        echo "[run.sh] Unknown mode: ${MODE}"
        usage
        ;;
esac
