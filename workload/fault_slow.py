import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phase2_common import write_metadata, append_label_csv

from workload import FaultInjection, run_workload

class FailSlow(FaultInjection):
    def __init__(self, rank, fault_rank, delay_seconds):
        super().__init__(rank, fault_rank)
        self.delay_seconds = delay_seconds

    def should_inject(self, iteration):
        # Fail-slow will be injected at every iteration to simulate a straggler
        return self.rank == self.fault_rank

    def inject(self):
        print(f"Rank {self.rank}: injecting delay of {self.delay_seconds} seconds", flush=True)
        time.sleep(self.delay_seconds)


def run_fault_slow(rank, world_size, fault_rank, delay_seconds, run_dir, label_csv, master_port):
    fault_injection = FailSlow(rank, fault_rank, delay_seconds)

    run_workload(
        rank=rank,
        world_size=world_size,
        num_iterations=5,
        workload_timout=timedelta(seconds=300),
        fault_injection=fault_injection,
        run_dir=run_dir,
        master_port=master_port
    )

def main():
    if len(sys.argv) != 7:
        print("Usage: python3 fault_slow.py <run_id> <run_dir> <fault_rank> <delay_seconds> <master_port> <label_csv>")
        sys.exit(1)

    run_id = sys.argv[1]
    run_dir = sys.argv[2]
    fault_rank = int(sys.argv[3])
    delay_seconds = int(sys.argv[4])
    master_port = sys.argv[5]
    label_csv = sys.argv[6]

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("Phase 2 requires at least 2 GPUs.")

    metadata = {
        "run_id": run_id,
        "scenario": "fail_slow",
        "fault_type": "gpu_straggler",
        "affected_rank": fault_rank,
        "delay_seconds": delay_seconds,
        "oom_injection": False,
        "world_size": world_size,
        "master_port": master_port,
        "status": "started"
    }
    write_metadata(run_dir, metadata)

    try:
        mp.spawn(
            run_fault_slow,
            args=(world_size, fault_rank, delay_seconds, run_dir, label_csv, master_port),
            nprocs=world_size,
            join=True
        )
        final_status = "completed"
    except Exception as e:
        print(f"Main caught exception: {e}", flush=True)
        final_status = "runtime_error"

    metadata["status"] = final_status
    write_metadata(run_dir, metadata)

    append_label_csv(label_csv, {
        "run_id": run_id,
        "scenario": "fail_slow",
        "fault_type": "gpu_straggler",
        "affected_rank": fault_rank,
        "delay_seconds": delay_seconds,
        "oom_injection": False,
        "world_size": world_size,
        "master_port": master_port,
        "status": final_status
    })

if __name__ == "__main__":
    main()
