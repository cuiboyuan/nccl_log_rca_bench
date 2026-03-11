from generic_workload import FaultInjection, run_workload

def run_normal(rank, world_size, num_iteration, workload_timeout, run_dir, label_csv, master_port):
    fault_injection = None

    run_workload(
        rank=rank,
        world_size=world_size,
        num_iterations=num_iteration,
        workload_timout=workload_timeout,
        fault_injection=None,
        run_dir=run_dir,
        master_port=master_port
    )
