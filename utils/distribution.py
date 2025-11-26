import os
import argparse
import torch
import torch.distributed as dist
from typing import Tuple

def init_distributed_mode(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.distributed or env_world_size > 1
    rank = 0
    world_size = max(env_world_size, 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not distributed:
        return False, rank, world_size, local_rank, device

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but distributed training was requested.")

    backend = "nccl"
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    cuda_available = torch.cuda.is_available()
    if backend == "nccl" and not cuda_available:
        backend = "gloo"

    if cuda_available and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif cuda_available and local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested local_rank={local_rank}, but only {torch.cuda.device_count()} GPU(s) are available."
            " Adjust the launch command (e.g., --nproc_per_node) or disable --distributed."
        )
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    args.distributed = True
    return True, rank, world_size, local_rank, device