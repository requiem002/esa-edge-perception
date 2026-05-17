#!/usr/bin/env bash
# Stop everything in the YOLO perception pipeline.
#   - inference.py inside the container
#   - stream_server.py inside the container
#   - lidar_stream.py on the host
#   - network_test.py on the host (and any leftover proxy listeners)
# Then print the status dashboard.
#
# Usage:
#   ./stop.sh

set -u

CONTAINER="yolo-saad"
DESKTOP="${HOME}/Desktop"

stop_in_container() {
    local pattern="$1"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        if docker exec "${CONTAINER}" pgrep -f "${pattern}" > /dev/null 2>&1; then
            echo "[stop.sh] Killing '${pattern}' inside ${CONTAINER}..."
            docker exec "${CONTAINER}" pkill -TERM -f "${pattern}" 2>/dev/null || true
            sleep 1
            docker exec "${CONTAINER}" pkill -KILL -f "${pattern}" 2>/dev/null || true
        else
            echo "[stop.sh] Not running in container: ${pattern}"
        fi
    else
        echo "[stop.sh] Container ${CONTAINER} not running — skipping ${pattern}"
    fi
}

stop_on_host() {
    local pattern="$1"
    if pgrep -f "${pattern}" > /dev/null 2>&1; then
        echo "[stop.sh] Killing '${pattern}' on host..."
        pkill -TERM -f "${pattern}" 2>/dev/null || true
        sleep 1
        pkill -KILL -f "${pattern}" 2>/dev/null || true
    else
        echo "[stop.sh] Not running on host: ${pattern}"
    fi
}

echo "[stop.sh] Stopping YOLO perception pipeline..."
echo

stop_in_container "/workspace/inference.py"
stop_in_container "/workspace/stream_server.py"
stop_on_host    "lidar_stream.py"
stop_on_host    "network_test.py"
stop_on_host    "stream_client.py"
stop_on_host    "dashboard_server.py"
stop_on_host    "sim_capture.py"
stop_on_host    "sim_camera_feed.py"
stop_on_host    "go2_api.py"

for pidfile in /tmp/dashboard.pid /tmp/sim_capture.pid /tmp/sim_cam.pid /tmp/go2_api.pid; do
    if [ -f "${pidfile}" ]; then
        kill "$(cat "${pidfile}")" 2>/dev/null || true
        rm "${pidfile}"
        echo "[stop.sh] Cleaned up ${pidfile}"
    fi
done

echo
echo "[stop.sh] Status:"
python3 "${DESKTOP}/src/robot/status.py" || true
