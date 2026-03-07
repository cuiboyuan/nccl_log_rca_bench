import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phase2_common import setup_nccl_env, write_metadata, append_label_csv

def run_fault_oom(rank, world_size, fault_rank, run_dir, label_csv, master_port):
    setup_nccl_env(run_dir, master_port)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60)
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"Rank {rank}: started on {device}", flush=True)

    try:
        # Step 1: all ranks do a small normal collective first
        tensor = torch.ones(10, device=device) * (rank + 1)
        print(f"Rank {rank}: before warmup all_reduce = {tensor}", flush=True)
        dist.all_reduce(tensor)
        print(f"Rank {rank}: after warmup all_reduce = {tensor}", flush=True)

        # Small sync point
        dist.barrier()

        # Step 2: only fault rank triggers OOM after NCCL is already active
        if rank == fault_rank:
            print(f"Rank {rank}: injecting delayed CUDA OOM", flush=True)
            time.sleep(3)

            allocations = []
            while True:
                # Allocate chunks gradually until OOM happens
                x = torch.empty((256, 1024, 1024), device=device)
                allocations.append(x)

        else:
            # Non-fault ranks attempt another collective and may hang/fail after peer dies
            tensor2 = torch.ones(10, device=device) * (rank + 10)
            print(f"Rank {rank}: before second all_reduce = {tensor2}", flush=True)
            dist.all_reduce(tensor2)
            print(f"Rank {rank}: after second all_reduce = {tensor2}", flush=True)

    except RuntimeError as e:
        print(f"Rank {rank}: RuntimeError: {e}", flush=True)
        raise

    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

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
