import os
import json
import socket
import time
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
