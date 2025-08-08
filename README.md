# AI System Service Management

## Services Overview

### unified-inference.service

Role: The shared AI model server. It runs vLLM to handle GPU-intensive inference requests from both the Log Analyzer and the AI Reviewer. This is the most critical service.

### log-analyzer-api.service

Role: The web server (Gunicorn/FastAPI) for the Log Analyzer. It exposes the public API for uploading logs and managing analysis jobs.

### log-analyzer-worker.service

Role: The background job processor (Celery) for the Log Analyzer. It performs the long-running analysis of log files.

### ai-reviewer-api.service

Role: The web server for the AI Code Reviewer. It serves the API for submitting code diffs and retrieving reviews.

### ai-reviewer-worker.service

Role: The background job processor for the AI Code Reviewer. It executes the code review tasks.

## Management Commands

Use these commands to control the state of the application services.

## ▶️ Start All Services

```bash
sudo systemctl start unified-inference.service log-analyzer-api.service log-analyzer-worker.service ai-reviewer-api.service ai-reviewer-worker.service
```

## ⏹️ Stop All Services

```bash
sudo systemctl stop unified-inference.service log-analyzer-api.service log-analyzer-worker.service ai-reviewer-api.service ai-reviewer-worker.service
```

## 📊 Check the Status of All Services

```bash
sudo systemctl status unified-inference.service log-analyzer-api.service log-analyzer-worker.service ai-reviewer-api.service ai-reviewer-worker.service
```

## 📜 View Real-Time Logs for a Specific Service

For the log analyzer worker:

```bash
journalctl -u log-analyzer-worker.service -f
```

For the code reviewer worker:

```bash
journalctl -u ai-reviewer-worker.service -f
```

For the shared AI model server:

```bash
journalctl -u unified-inference.service -f
```

## 🧹 Emergency Cleanup: Kill Stray Processes

If a restart command fails or you suspect "zombie" processes are running, use pkill to forcefully stop all related processes before attempting another restart.

```bash
sudo pkill -9 -f gunicorn
sudo pkill -9 -f celery
echo "All stray gunicorn and celery processes have been killed."
```
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
