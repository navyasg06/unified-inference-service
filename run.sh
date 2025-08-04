#!/bin/bash

# run.sh
# This version uses process groups, a standard Linux feature,
# to ensure all child processes are cleaned up.

# Exit immediately if a command exits with a non-zero status.
set -e

# The 'trap' command catches signals and calls the 'cleanup' function.
trap 'cleanup' SIGINT SIGTERM

# Get the directory of the script to reliably activate the venv
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate the virtual environment
source "$DIR/venv/bin/activate"

echo "Starting Uvicorn server for unified-inference in a new process group..."

# Start Uvicorn in the background. The '&' is crucial.
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Store the PID of the Uvicorn process
UVICORN_PID=$!
echo "Uvicorn server started with PID: $UVICORN_PID"

cleanup() {
    echo "Caught signal! Cleaning up processes..."
    
    # pkill with the -P flag kills a process and all of its descendants.
    # This is the most reliable way to clean up vLLM's workers.
    # We send SIGKILL (-9) to ensure they die immediately and release the GPU.
    pkill -9 -P $UVICORN_PID
    
    echo "Killing main Uvicorn process: $UVICORN_PID"
    kill -9 $UVICORN_PID
    
    echo "Cleanup complete."
    exit 0
}

# The 'wait' command pauses the script here, allowing it to run
# indefinitely until the background Uvicorn process exits or a
# signal is received.
wait $UVICORN_PID