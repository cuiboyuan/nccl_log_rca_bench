import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phase2_common import write_metadata, append_label_csv

from workload import FaultInjection, run_workload

class FailStop(FaultInjection):
    def __init__(self, rank, fault_rank, fault_iteration):
        super().__init__(rank, fault_rank)
        self.fault_iteration = fault_iteration

    def should_inject(self, iteration):
        return self.rank == self.fault_rank and iteration == self.fault_iteration

    def inject(self):
        # Small sync point
        dist.barrier()
        # only fault rank triggers OOM after NCCL is already active
        print(f"Rank {self.rank}: injecting delayed CUDA OOM", flush=True)
        time.sleep(3)
        allocations = []
        while True:
            # Allocate chunks gradually until OOM happens
            x = torch.empty((256, 1024, 1024), device=device)
            allocations.append(x)


def run_fault_oom(rank, world_size, fault_rank, run_dir, label_csv, master_port):
    fault_injection = FailStop(rank, fault_rank, fault_iteration=0)

    run_workload(
        rank=rank,
        world_size=world_size,
        num_iterations=5,
        workload_timout=timedelta(seconds=120),
        fault_injection=fault_injection,
        run_dir=run_dir,
        master_port=master_port
    )

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 fault_oom.py <run_id> <run_dir> <fault_rank> <master_port> <label_csv>")
        sys.exit(1)

    run_id = sys.argv[1]
    run_dir = sys.argv[2]
    fault_rank = int(sys.argv[3])
    master_port = sys.argv[4]
    label_csv = sys.argv[5]

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("Phase 2 requires at least 2 GPUs.")

    metadata = {
        "run_id": run_id,
        "scenario": "fail_stop",
        "fault_type": "cuda_oom",
        "affected_rank": fault_rank,
        "delay_seconds": None,
        "oom_injection": True,
        "world_size": world_size,
        "master_port": master_port,
        "status": "started"
    }
    write_metadata(run_dir, metadata)

    try:
        mp.spawn(
            run_fault_oom,
            args=(world_size, fault_rank, run_dir, label_csv, master_port),
            nprocs=world_size,
            join=True
        )
        final_status = "completed"
    except Exception as e:
        print(f"Main caught exception: {e}", flush=True)
        final_status = "fault_triggered_or_failed"

    metadata["status"] = final_status
    write_metadata(run_dir, metadata)

    append_label_csv(label_csv, {
        "run_id": run_id,
        "scenario": "fail_stop",
        "fault_type": "cuda_oom",
        "affected_rank": fault_rank,
        "delay_seconds": "",
        "oom_injection": True,
        "world_size": world_size,
        "master_port": master_port,
        "status": final_status
    })

if __name__ == "__main__":
    main()
