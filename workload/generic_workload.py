import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phase2_common import setup_nccl_env, write_metadata, append_label_csv

class FaultInjection:
    def __init__(self, rank, fault_rank):
        self.rank = rank
        self.fault_rank = fault_rank

    def should_inject(self, iteration):
        raise NotImplementedError("should_inject() must be implemented by subclasses")

    def inject(self):
        raise NotImplementedError("inject() must be implemented by subclasses")


def run_workload(rank, world_size, num_iterations, workload_timout, fault_injection, run_dir, master_port):
    setup_nccl_env(run_dir, master_port)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
        timeout=timedelta(seconds=300) # per-operation timeout to prevent hanging indefinitely on NCCL collectives
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"Rank {rank}: started on {device}", flush=True)

    # All ranks do a small normal collective first
    tensor = torch.ones(10, device=device) * (rank + 1)
    print(f"Rank {rank}: before warmup all_reduce = {tensor}", flush=True)
    dist.all_reduce(tensor)
    print(f"Rank {rank}: after warmup all_reduce = {tensor}", flush=True)

    try:
        # Example workload: num_iterations of all_reduce with some computation in between
        for i in range(num_iterations):
            print(f"Rank {rank}: ==== Iteration {i+1}/{num_iterations} ====")
            # Simulate some computation
            tensor = torch.ones(10, device=device) * (rank + 1 + i)
            time.sleep(1)

            # Fault injection
            if fault_injection is None:
                # Normal scenario with no faults
                pass
            elif fault_injection.should_inject(i):
                fault_injection.inject()

            # Collective operation to synchronize
            print(f"Rank {rank}: before all_reduce iter {i} = {tensor}", flush=True)
            dist.all_reduce(tensor)
            print(f"Rank {rank}: after all_reduce iter {i} = {tensor}", flush=True)

    except RuntimeError as e:
        print(f"Rank {rank}: RuntimeError: {e}", flush=True)
        raise

    finally:
        try:
            print(f"Rank {rank}: End of workload.")
            dist.destroy_process_group()
        except Exception as e:
            print(f"Rank {rank}: Exception during destroy_process_group: {e}", flush=True)
            raise

