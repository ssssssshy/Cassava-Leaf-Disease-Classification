import os
import torch
import torch.distributed as dist


def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: float, world_size: int, device: torch.device) -> float:

    if not is_distributed():
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()
