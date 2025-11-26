import os
from typing import Any, Dict, Optional
import torch
from src.ARC_ViT import ARCViT  
from src.ARC_UNet import ARCUNet
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from utils.lr_scheduler import get_cosine_schedule_with_warmup

def count_parameters(model):
    ret = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'task_token' not in name:
            #print(name, p.numel())
            ret += p.numel()
    return ret

def get_model_arch(args, train_dataset):
    if args.architecture == "vit":
        model = ARCViT(
            num_tasks=train_dataset.num_tasks,
            image_size=args.image_size,
            num_colors=args.num_colors,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            patch_size=args.patch_size,
        )
    else:
        model = ARCUNet(
            num_tasks=train_dataset.num_tasks,
            image_size=args.image_size,
            num_colors=args.num_colors,
            size=args.unet_size,
        )

    return model


# Resume from checkpoint if specified
def load_models(args, train_dataset, device, distributed, rank, local_rank):
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    resume_reset_epoch = bool(getattr(args, "resume_reset_epoch", False))
    start_epoch = 1
    checkpoint: Optional[Dict[str, Any]] = None
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model = get_model_arch(args, train_dataset)
    
        state_dict = checkpoint.get("model_state", {})
        state_dict = {
            key.replace("_orig_mod.", "", 1): value
            for key, value in checkpoint["model_state"].items()
        }
        if args.resume_skip_task_token and "task_token_embed.weight" in state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k != "task_token_embed.weight"}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Skipped loading parameters: {sorted(missing)}")
            if unexpected:
                print(f"Unexpected parameters ignored from checkpoint: {sorted(unexpected)}")
        else:
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as exc:
                checkpoint_weight = state_dict.get("task_token_embed.weight")
                current_weight = model.task_token_embed.weight
                if (
                    checkpoint_weight is not None
                    and checkpoint_weight.shape != current_weight.shape
                    and not args.resume_skip_task_token
                ):
                    raise RuntimeError(
                        "Mismatch in task_token_embed.weight shape. "
                        "Re-run with --resume-skip-task-token to reuse other weights."
                    ) from exc
                raise
        model.to(device)
        
        # Apply torch.compile for speedup (PyTorch 2.0+)
        if not args.no_compile and hasattr(torch, 'compile'):
            if (not distributed) or rank == 0:
                print("Applying torch.compile for optimization...")
            model = torch.compile(model, mode=args.compile_mode)
        
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )
        model_for_eval = model.module if distributed else model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        # Optionally restore epoch
        if "epoch" in checkpoint and not resume_reset_epoch:
            start_epoch = checkpoint["epoch"] + 1
        elif "epoch" in checkpoint and resume_reset_epoch and ((not distributed) or rank == 0):
            print("Ignoring checkpoint epoch due to --resume-reset-epoch; restarting from epoch 1.")
    else:
        model = get_model_arch(args, train_dataset)
      
        print(f"Parameter count: {count_parameters(model) / 1_000_000:.2f}M (excluding task tokens)")
        model.to(device)
        
        # Apply torch.compile for speedup (PyTorch 2.0+)
        if not args.no_compile and hasattr(torch, 'compile'):
            if (not distributed) or rank == 0:
                print("Applying torch.compile for optimization...")
            model = torch.compile(model, mode=args.compile_mode)
        
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )
        model_for_eval = model.module if distributed else model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    if (not distributed) or rank == 0:
        if scaler.is_enabled():
            print("Using automatic mixed precision (AMP) training")
        else:
            print("AMP disabled (CPU or --no-amp flag)")

    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, min(args.epochs // 5, 10), args.epochs)
    else:
        scheduler = None
    return model, model_for_eval, optimizer, scaler, scheduler, start_epoch



def load_optimizer(args, model, device, distributed, rank):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    if (not distributed) or rank == 0:
        if scaler.is_enabled():
            print("Using automatic mixed precision (AMP) training")
        else:
            print("AMP disabled (CPU or --no-amp flag)")
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, min(args.epochs // 5, 10), args.epochs)
    else:
        scheduler = None

    return optimizer, scaler, scheduler

def load_model_only(args, train_dataset, device, distributed, rank, local_rank):
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model = get_model_arch(args, train_dataset)
    
        state_dict = checkpoint.get("model_state", {})
        state_dict = {
            key.replace("_orig_mod.", "", 1): value
            for key, value in checkpoint["model_state"].items()
        }
        if args.resume_skip_task_token and "task_token_embed.weight" in state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k != "task_token_embed.weight"}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Skipped loading parameters: {sorted(missing)}")
            if unexpected:
                print(f"Unexpected parameters ignored from checkpoint: {sorted(unexpected)}")
        else:
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as exc:
                checkpoint_weight = state_dict.get("task_token_embed.weight")
                current_weight = model.task_token_embed.weight
                if (
                    checkpoint_weight is not None
                    and checkpoint_weight.shape != current_weight.shape
                    and not args.resume_skip_task_token
                ):
                    raise RuntimeError(
                        "Mismatch in task_token_embed.weight shape. "
                        "Re-run with --resume-skip-task-token to reuse other weights."
                    ) from exc
                raise
        model.to(device)
        
        # Apply torch.compile for speedup (PyTorch 2.0+)
        if not args.no_compile and hasattr(torch, 'compile'):
            if (not distributed) or rank == 0:
                print("Applying torch.compile for optimization...")
            model = torch.compile(model, mode=args.compile_mode)
        
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )
    return model