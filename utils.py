import torch
import math
from einops import rearrange
import random

def linear_warmup_cosine_decay(optimizer, warmup_steps, total_steps):
    """
    Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps
    """

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, fn
    )

def lr_search(optimizer):
    """
    LR Search for finding the upper bound
    """

    def fn(step):
        return (step * 1e-8) * (step * 1.001)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, fn
    )
    
def get_cycles_buildoff(
    optimizer, num_warmup_steps: int, num_training_steps: int, noise_amount: float = 0.0, num_cycles: int = 10, merge_cycles: int = 4, last_epoch: int = -1
):
    random_state = random.Random(94839)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        cycle_progress = float(num_cycles) * progress
        start_cycles = num_cycles - merge_cycles
        step_noise = noise_amount
        if cycle_progress > start_cycles:
            build_down_cycles = cycle_progress - start_cycles
            cycle_progress = start_cycles + (build_down_cycles / merge_cycles)
            # During buildoff there will be no noise
            step_noise = 0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (cycle_progress % 1.0)))) + (random_state.uniform(-1, 1) * step_noise)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
