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
        --stream)         MODE="stream"     ; shift ;;
        --status)         MODE="status"     ; shift ;;
        --experiment)     MODE="experiment" ; shift ;;
        --no-display)     MODE="inference"  ; INFERENCE_EXTRA+=("$1") ; shift ;;
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

    *)
        echo "[run.sh] Unknown mode: ${MODE}"
        usage
        ;;
esac
