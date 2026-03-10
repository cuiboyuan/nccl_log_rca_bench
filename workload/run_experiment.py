import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phase2_common import write_metadata, append_label_csv
from fault_slow import run_fault_slow
from fault_oom import run_fault_oom

def main():
    run_id = sys.argv[1]
    run_dir = sys.argv[2]
    fault_rank = int(sys.argv[3])
    master_port = sys.argv[4]
    label_csv = sys.argv[5]
    fault_scenario = sys.argv[6]

    if fault_scenario == "fail_slow":
        delay_seconds = int(sys.argv[7])
        fault_type = "gpu_straggler"
        oom_injection = False
    elif fault_scenario == "fail_stop":
        delay_seconds = None
        fault_type = "cuda_oom"
        oom_injection = True
    else:
        print(f"Unknown fault scenario: {fault_scenario}")
        sys.exit(1)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("To requires at least 2 GPUs.")

    metadata = {
        "run_id": run_id,
        "scenario": fault_scenario,
        "fault_type": fault_type,
        "affected_rank": fault_rank,
        "delay_seconds": delay_seconds,
        "oom_injection": oom_injection,
        "world_size": world_size,
        "master_port": master_port,
        "status": "started"
    }
    write_metadata(run_dir, metadata)

    if fault_scenario == "fail_slow":
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

    elif fault_scenario == "fail_stop":
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
        "scenario": fault_scenario,
        "fault_type": fault_type,
        "affected_rank": fault_rank,
        "delay_seconds": delay_seconds,
        "oom_injection": oom_injection,
        "world_size": world_size,
        "master_port": master_port,
        "status": final_status
    })


if __name__ == "__main__":
    main()