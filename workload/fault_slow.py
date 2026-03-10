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
        workload_timout=300,
        fault_injection=fault_injection,
        run_dir=run_dir,
        master_port=master_port
    )
