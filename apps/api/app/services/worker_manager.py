"""Local Worker Process Management for Stream Ingestion and Inference.

Isolates heavy GPU / CV execution from FastAPI while providing lifecycle control:
STARTING -> RUNNING -> STOPPED / FAILED
"""
import logging
import os
import subprocess
import sys
from typing import Dict, Optional, Tuple

logger = logging.getLogger("emergency_vision.api.worker_manager")


class WorkerProcessInfo:
    """Metadata tracking a running worker subprocess."""

    def __init__(self, process: subprocess.Popen, stream_id: str, source_url: str):
        self.process = process
        self.stream_id = stream_id
        self.source_url = source_url
        self.pid = process.pid


class LocalWorkerManager:
    """Manages spawning, monitoring, and terminating CV worker processes."""

    def __init__(self) -> None:
        self._workers: Dict[str, WorkerProcessInfo] = {}

    def start_worker(
        self,
        stream_id: str,
        source_url: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        line_ratio: Optional[float] = None,
        publisher: str = "http",
        api_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        redis_stream: Optional[str] = None,
    ) -> int:
        """Start a dedicated worker subprocess for the stream.

        Returns:
            Process ID (PID) of the spawned worker.
        """
        # Stop existing worker for the same stream_id if any
        if stream_id in self._workers:
            self.stop_worker(stream_id)

        cmd = [
            sys.executable,
            "-m",
            "apps.worker.app.main",
            "--source",
            str(source_url),
            "--stream-id",
            str(stream_id),
            "--publisher",
            str(publisher or "http"),
        ]
        if api_url:
            cmd.extend(["--api-url", api_url])
        if redis_url:
            cmd.extend(["--redis-url", redis_url])
        if redis_stream:
            cmd.extend(["--redis-stream", redis_stream])
        if model_path:
            cmd.extend(["--model", model_path])
        if device:
            cmd.extend(["--device", device])
        if output_path:
            cmd.extend(["--output", output_path])
        if max_frames:
            cmd.extend(["--max-frames", str(max_frames)])

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()

        logger.info("Spawning worker for stream %s: %s", stream_id, " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._workers[stream_id] = WorkerProcessInfo(process, stream_id, source_url)
        return process.pid

    def check_worker_status(self, stream_id: str) -> Tuple[str, Optional[int], Optional[str]]:
        """Check status of worker subprocess.

        Returns:
            Tuple of (status_string, returncode, error_message)
            Status is one of: "running", "stopped", "failed"
        """
        worker_info = self._workers.get(stream_id)
        if not worker_info:
            return "stopped", None, None

        ret = worker_info.process.poll()
        if ret is None:
            return "running", None, None
        elif ret == 0:
            return "stopped", 0, None
        else:
            stderr_snippet = ""
            if worker_info.process.stderr:
                try:
                    stderr_bytes = worker_info.process.stderr.read()
                    if stderr_bytes:
                        stderr_snippet = stderr_bytes.decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
            error_msg = f"Worker exited with code {ret}"
            if stderr_snippet:
                error_msg = f"{error_msg}: {stderr_snippet.strip()}"
            return "failed", ret, error_msg

    def stop_worker(self, stream_id: str) -> bool:
        """Stop worker subprocess gracefully, killing if needed."""
        worker_info = self._workers.get(stream_id)
        if not worker_info:
            return False

        process = worker_info.process
        if process.poll() is None:
            logger.info("Terminating worker PID %d for stream %s", process.pid, stream_id)
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning("Worker PID %d did not terminate cleanly; killing...", process.pid)
                process.kill()
                try:
                    process.wait(timeout=1)
                except Exception:
                    pass

        return True

    def cleanup_all(self) -> None:
        """Stop all running workers on server shutdown."""
        for stream_id in list(self._workers.keys()):
            self.stop_worker(stream_id)
        self._workers.clear()
