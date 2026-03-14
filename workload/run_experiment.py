import sys
import time
import threading
from torch.multiprocessing.spawn import ProcessExitedException, ProcessRaisedException
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phase2_common import write_metadata, append_label_csv
from fault_slow import run_fault_slow
from fault_oom import run_fault_oom
from no_fault import run_normal

def main():
    run_id = sys.argv[1]
    run_dir = sys.argv[2]
    fault_rank = int(sys.argv[3])
    master_port = sys.argv[4]
    label_csv = sys.argv[5]
    fault_scenario = sys.argv[6]
    num_iteration = int(sys.argv[7])
    workload_timeout = int(sys.argv[8])

    if fault_scenario == "fail_slow":
        delay_seconds = int(sys.argv[9])
        injected_interation = "all"
        fault_type = "gpu_straggler"
        oom_injection = False
    elif fault_scenario == "fail_stop":
        delay_seconds = None
        injected_interation = int(sys.argv[9])
        fault_type = "cuda_oom"
        oom_injection = True
    elif fault_scenario == "normal":
        delay_seconds = None
        injected_interation = None
        fault_type = "no_fault"
        oom_injection = False
    else:
        print(f"Unknown fault scenario: {fault_scenario}")
        sys.exit(1)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("Experiment requires at least 2 GPUs.")

    # Print version information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"NCCL version: {torch.cuda.nccl.version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of available GPUs: {world_size}")

    metadata = {
        "run_id": run_id,
        "scenario": fault_scenario,
        "fault_type": fault_type,
        "affected_rank": fault_rank,
        "delay_seconds": delay_seconds,
        "oom_injection": oom_injection,
        "world_size": world_size,
        "master_port": master_port,
        "fault_iteration": injected_interation,
        "total_iteration": num_iteration,
        "workload_timeout": workload_timeout,
        "status": "started"
    }
    write_metadata(run_dir, metadata)

    start = time.time()
    if fault_scenario == "fail_slow":
        ctx = mp.spawn(
            run_fault_slow,
            args=(world_size, num_iteration, workload_timeout, fault_rank, delay_seconds, run_dir, label_csv, master_port),
            nprocs=world_size,
            join=False
        )

    elif fault_scenario == "fail_stop":
        ctx = mp.spawn(
            run_fault_oom,
            args=(world_size, num_iteration, workload_timeout, fault_rank, run_dir, label_csv, master_port),
            nprocs=world_size,
            join=False
        )

    elif fault_scenario == "normal":
        ctx = mp.spawn(
            run_normal,
            args=(world_size, num_iteration, workload_timeout, run_dir, label_csv, master_port),
            nprocs=world_size,
            join=False
        )

    # Terminate any remaining processes if we hit the workload timeout
    timed_out_event = threading.Event()
    def kill_all():
        timed_out_event.set()
        print("Global timeout reached, terminating all processes.")
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()

    timer = threading.Timer(workload_timeout, kill_all)
    timer.start()

    try:
        while not ctx.join(timeout=5):  # poll every 5s
            pass
        final_status = "completed"
    except ProcessExitedException as e:
        if timed_out_event.is_set():
            print(f"Run timed out after {workload_timeout}s — processes terminated cleanly.")
            final_status = "timeout"
        else:
            print(f"A process died unexpectedly: {e}")
            final_status = "failed"
    except ProcessRaisedException as e:
        if "OutOfMemoryError" in str(e) or "out of memory" in str(e).lower():
            print(f"A process raised an OOM exception: {e}")
            final_status = "oom_fault_triggered"
        else:
            print(f"A process raised an exception: {e}")
            final_status = "failed"
    finally:
        timer.cancel()
    total_time = time.time() - start
    print(f"Total elapsed time: {total_time:.2f} seconds")

    metadata["status"] = final_status
    write_metadata(run_dir, metadata)

    append_label_csv(label_csv, {
        "run_id": run_id,
        "scenario": fault_scenario,
        "fault_type": fault_type,
        "affected_rank": fault_rank,
        "delay_seconds": delay_seconds,
        "oom_injection": oom_injection,
        "world_size": world_size,
        "master_port": master_port,
        "fault_iteration": injected_interation,
        "total_iteration": num_iteration,
        "workload_timeout": workload_timeout,
        "status": final_status
    })


if __name__ == "__main__":
    main()