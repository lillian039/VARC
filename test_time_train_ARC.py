import argparse
from copy import deepcopy
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from utils.args import parse_args
from utils.distribution import init_distributed_mode
from utils.load_model import load_model_only, load_optimizer

from src.ARC_loader import build_dataloaders, IGNORE_INDEX
from utils.eval_utils_ttt import generate_predictions, get_eval_rot_transform_resolver


def _format_eta(seconds: float) -> str:
    total_seconds = int(max(seconds, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ttt_once(model, device, distributed, rank, train_loader, train_sampler, eval_loader, cur_attempt_idx):
    autocast_device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cuda"
    is_main_process = (not distributed) or rank == 0

    global_start = time.time()
    previous_total_steps = 0
    optimizer, scaler, scheduler = load_optimizer(
        model=model, args=args, device=device, distributed=distributed, rank=rank
    )
    try:
        for epoch in range(0, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            sample_count = 0
            total_batches = len(train_loader)
            epoch_start = time.time()
            train_exact = 0
            train_examples = 0

            for step, batch in enumerate(train_loader, 1):
                inputs = batch["inputs"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                task_ids = batch["task_ids"].to(device)

                optimizer.zero_grad(set_to_none=True)
                
                # Use automatic mixed precision
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    logits = model(inputs, task_ids, attention_mask=attention_mask)
                    num_colors = logits.size(1)
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
                    loss = F.cross_entropy(
                        logits_flat,
                        targets.view(-1),
                        ignore_index=IGNORE_INDEX,
                    )

                batch_size = inputs.size(0)
                predictions = logits.argmax(dim=1)
                for idx in range(batch_size):
                    target = targets[idx]
                    prediction = predictions[idx]
                    valid = target != IGNORE_INDEX
                    if valid.any():
                        is_exact = bool(torch.equal(prediction[valid], target[valid]))
                    else:
                        is_exact = False
                    train_exact += int(is_exact)
                    train_examples += 1

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * batch_size
                sample_count += batch_size

                if total_batches > 0 and is_main_process and step % 10 == 0:  # Update every 10 steps
                    elapsed = time.time() - epoch_start
                    avg_step_time = elapsed / step
                    steps_completed = previous_total_steps + step
                    total_steps = len(train_loader) * args.epochs
                    remaining_steps = total_steps - steps_completed
                    elapsed_global = time.time() - global_start
                    avg_time_per_step_global = elapsed_global / max(steps_completed, 1)
                    eta = remaining_steps * avg_time_per_step_global
                    bar_length = 30
                    progress_ratio = steps_completed / total_steps if total_steps else 0
                    filled = int(bar_length * progress_ratio)
                    bar = "#" * filled + "-" * (bar_length - filled)
                    progress = 100.0 * progress_ratio
                    sys.stdout.write(
                        f"\rEpoch {epoch} [{bar}] {progress:5.1f}% ETA {_format_eta(eta)}"
                    )
                    sys.stdout.flush()

            if total_batches > 0 and is_main_process:
                sys.stdout.write("\n")
            previous_total_steps += total_batches

            epoch_duration = time.time() - epoch_start if total_batches > 0 else 0.0

            train_totals = torch.tensor(
                [running_loss, sample_count, train_exact, train_examples],
                dtype=torch.float64,
                device=device,
            )
            if distributed and dist.is_initialized():
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            running_loss_total, sample_count_total, train_exact_total, train_examples_total = train_totals.tolist()
            avg_train_loss = running_loss_total / max(sample_count_total, 1)
            train_acc = train_exact_total / max(train_examples_total, 1)

            total_elapsed = time.time() - global_start
            total_steps = len(train_loader) * args.epochs
            steps_completed = min(previous_total_steps, total_steps)
            remaining_steps = total_steps - steps_completed
            avg_time_per_step_global = total_elapsed / max(steps_completed, 1)
            total_eta = remaining_steps * avg_time_per_step_global

            log_parts = [
                f"epoch={epoch}",
                f"train_loss={avg_train_loss:.4f}",
                f"train_acc={train_acc:.4f}",
                f"epoch_time={epoch_duration:.1f}s",
                f"eta_total={_format_eta(total_eta)}",
            ]

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.learning_rate
            log_parts.append(f"lr={current_lr:.6f}")           
            if is_main_process:
                print(" | ".join(log_parts))

            if scheduler is not None:
                scheduler.step()

    finally:
        if distributed and dist.is_initialized():
            dist.barrier()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    generate_predictions(
        model,
        eval_loader,
        device,
        img_size=args.image_size,
        attempt_nums=args.num_attempts,
        task_transform_resolver=get_eval_rot_transform_resolver(),
        fix_scale_factor=args.fix_scale_factor,
        disable_translation=args.disable_translation,
        if_fix_scale=args.disable_resolution_augmentation,
        save_name=args.eval_save_name + "_attempt_" + str(cur_attempt_idx),
        eval_split=args.eval_split,
        task_type=args.data_root.split("/")[-1],  # e.g., "ARC-AGI"
    )

def train(args: argparse.Namespace) -> None:
    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    set_seed(args.seed + (rank if distributed else 0))

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    total_train_examples = len(train_dataset)

    if (not distributed) or rank == 0:
        print(f"Total training examples: {total_train_examples}")


    model_original = load_model_only(
        args=args, train_dataset=train_dataset, device=device, distributed=distributed, rank=rank, local_rank=local_rank
    )
    
    for attempt_idx in range(args.ttt_num_each):
        model = deepcopy(model_original)
        print(f"Starting test-time training attempt {attempt_idx + 1}/{args.ttt_num_each}...")
        ttt_once(model=model, device=device, distributed=distributed, rank=rank,
                train_loader=train_loader, train_sampler=train_sampler,
                eval_loader=eval_loader, cur_attempt_idx=attempt_idx)



if __name__ == "__main__":
    args = parse_args()
    train(args)
