import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phase2_common import setup_nccl_env, write_metadata, append_label_csv

def run_fault_slow(rank, world_size, fault_rank, delay_seconds, run_dir, label_csv, master_port):
    setup_nccl_env(run_dir, master_port)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90)
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    tensor = torch.ones(10, device=device) * (rank + 1)

    print(f"Rank {rank}: started on {device}", flush=True)

    if rank == fault_rank:
        print(f"Rank {rank}: injecting delay of {delay_seconds} seconds", flush=True)
        time.sleep(delay_seconds)

    print(f"Rank {rank}: before all_reduce = {tensor}", flush=True)
    dist.all_reduce(tensor)
    print(f"Rank {rank}: after all_reduce = {tensor}", flush=True)

    dist.destroy_process_group()

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
