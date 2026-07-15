import os
import json
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

def setup_nccl_env(run_dir: str, master_port: str):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "TRACE"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ["NCCL_DEBUG_FILE"] = str(Path(run_dir) / "nccl_logs_%h_%p.txt")

def write_metadata(run_dir: str, metadata: dict):
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    with open(run_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def append_label_csv(label_csv: str, row: dict):
    path = Path(label_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "run_id",
        "scenario",
        "fault_type",
        "affected_rank",
        "delay_seconds",
        "oom_injection",
        "world_size",
        "master_port",
        "fault_iteration",
        "total_iteration",
        "workload_timeout",
        "status"
    ]

    write_header = not path.exists()
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(columns) + "\n")
        values = [str(row.get(c, "")) for c in columns]
        f.write(",".join(values) + "\n")

def get_hostname():
    return socket.gethostname()

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

def resolve_nccl_log_path(run_dir: str) -> Path:
    """Return the NCCL log file path NCCL will write for the current process."""
    hostname = socket.gethostname()
    pid = os.getpid()
    return Path(run_dir) / f"nccl_logs_{hostname}_{pid}.txt"


class LineTimestampTracker:
    """Background thread that polls a NCCL log file every second and records
    the Unix timestamp at which each line first appeared."""

    def __init__(self, log_path: Path, out_path: Path):
        self.log_path = log_path
        self.out_path = out_path
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._timestamps: list = []
        self._start_iso: str = ""

    def start(self):
        self._start_iso = datetime.now().isoformat()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self._flush()
        self._write_output()

    def _poll(self):
        """Read the log file and append timestamps for any newly seen lines.
        Writes the output file after every update so data survives SIGTERM."""
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path, "r", errors="replace") as f:
                count = sum(1 for _ in f)
        except OSError:
            return
        if count > len(self._timestamps):
            ts = time.time()
            for _ in range(count - len(self._timestamps)):
                self._timestamps.append(ts)
            self._write_output()

    def _run(self):
        while not self._stop_event.wait(1.0):
            self._poll()

    def _flush(self):
        """Final poll to capture lines written after the last scheduled tick."""
        self._poll()

    def _write_output(self):
        """Atomically write the timestamp JSON (temp file + rename)."""
        tmp = self.out_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({
                "start_iso": self._start_iso,
                "line_timestamps": self._timestamps,
            }, f)
        tmp.replace(self.out_path)
