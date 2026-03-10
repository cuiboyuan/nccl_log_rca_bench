from generic_workload import FaultInjection, run_workload

def run_normal(rank, world_size, run_dir, label_csv, master_port):
    fault_injection = None

    run_workload(
        rank=rank,
        world_size=world_size,
        num_iterations=5,
        workload_timout=300,
        fault_injection=None,
        run_dir=run_dir,
        master_port=master_port
    )
