import os

import torch
import torch.distributed as dist


def setup_distributed_gloo(backend="gloo", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size


def setup_distributed_nccl(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # for GPU memory balance
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size
