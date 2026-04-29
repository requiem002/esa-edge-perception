#!/usr/bin/env bash
# Launch YOLO pedestrian detection inside the yolo-saad container.
# Usage: ./run.sh [extra args for inference.py]
#
# Examples:
#   ./run.sh                          # defaults: engine model, webcam, conf=0.25
#   ./run.sh --conf 0.4               # higher confidence threshold
#   ./run.sh --model /workspace/yolo26n.pt   # force PyTorch model
#   ./run.sh --no-display             # headless (no GUI window)
#   ./run.sh --source /workspace/test.mp4    # run on video file
#   ./run.sh --source 2                      # MuJoCo simulator via v4l2loopback

set -euo pipefail

CONTAINER="yolo-saad"

# Check container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[ERROR] Container '${CONTAINER}' is not running."
    echo "Start it with:"
    echo "  docker run -d --name ${CONTAINER} --ipc=host --runtime nvidia --network host \\"
    echo "    -e DISPLAY=\$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \\"
    echo "    --device /dev/video0 --device /dev/video2 \\"
    echo "    -v ~/Desktop:/workspace yolo-orin:v2 sleep infinity"
    exit 1
fi

# Allow container to access X display
xhost +local: > /dev/null 2>&1 || true

echo "Starting pedestrian detection in container '${CONTAINER}'..."
echo "Press Ctrl+C or 'q' in the video window to stop."
echo ""

docker exec -it \
    -e DISPLAY="$DISPLAY" \
    "${CONTAINER}" \
    python3 /workspace/inference.py "$@"
