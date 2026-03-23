import time
import torch
import torch.distributed as dist

from generic_workload import FaultInjection, run_workload

class FailStop(FaultInjection):
    def __init__(self, rank, fault_rank, fault_iteration):
        super().__init__(rank, fault_rank)
        self.fault_iteration = fault_iteration
        self.device = torch.device(f"cuda:{rank}")

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
            x = torch.empty((256, 1024, 1024), device=self.device)
            allocations.append(x)


def run_fault_oom(rank, world_size, num_iteration, workload_timeout, fault_rank, injected_interation, run_dir, label_csv, master_port):
    fault_injection = FailStop(rank, fault_rank, fault_iteration=injected_interation)

    run_workload(
        rank=rank,
        world_size=world_size,
        num_iterations=num_iteration,
        workload_timout=workload_timeout,
        fault_injection=fault_injection,
        run_dir=run_dir,
        master_port=master_port
    )
