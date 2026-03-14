import time

from generic_workload import FaultInjection, run_workload

class FailSlow(FaultInjection):
    def __init__(self, rank, fault_rank, delay_seconds):
        super().__init__(rank, fault_rank)
        self.delay_seconds = delay_seconds

    def should_inject(self, iteration):
        # Fail-slow will be injected at every iteration to simulate a straggler
        return self.rank == self.fault_rank

    def inject(self):
        print(f"Rank {self.rank}: injecting delay of {self.delay_seconds} seconds", flush=True)
        for _ in range(self.delay_seconds):
            time.sleep(1)


def run_fault_slow(rank, world_size, num_iteration, workload_timeout, fault_rank, delay_seconds, run_dir, label_csv, master_port):
    fault_injection = FailSlow(rank, fault_rank, delay_seconds)

    run_workload(
        rank=rank,
        world_size=world_size,
        num_iterations=num_iteration,
        workload_timout=workload_timeout,
        fault_injection=fault_injection,
        run_dir=run_dir,
        master_port=master_port
    )
