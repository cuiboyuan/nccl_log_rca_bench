import torch
import os
from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp


def run_distributed_training(rank, world_size):
    """
    Simple distributed training example with all_reduce operation.

    Args:
        rank: Current process rank
        world_size: Total number of processes
    """
    # Initialize the process group
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29501')

    os.environ.setdefault('NCCL_P2P_DISABLE', '1')  # Disable P2P
    # os.environ.setdefault('NCCL_IB_DISABLE', '1')  # Disable InfiniBand

    os.environ.setdefault('NCCL_DEBUG', 'TRACE')  # Enable detailed NCCL debug output
    os.environ.setdefault('NCCL_DEBUG_SUBSYS', 'ALL')  # Log all NCCL subsystems
    os.environ.setdefault('NCCL_DEBUG_FILE', 'nccl_logs_%h_%p.txt')  # Write debug logs to file

    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                           timeout=timedelta(seconds=30)) # 30s timeout

    # Set the device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"Rank {rank}: Starting on device {device}")
    print(f"Rank {rank}: CUDA Device: {torch.cuda.get_device_name(rank)}")

    # Create a tensor on the GPU
    tensor = torch.ones(10) * (rank + 1)
    tensor = tensor.to(device)

    print(f"Rank {rank}: Before all_reduce: {tensor}")

    # Perform all_reduce operation (sum by default)
    dist.all_reduce(tensor)

    print(f"Rank {rank}: After all_reduce (sum): {tensor}")

    # Cleanup
    dist.destroy_process_group()


def main():
    # Print version information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"NCCL version: {torch.cuda.nccl.version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")

    mp.spawn(run_distributed_training, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()