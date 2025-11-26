import argparse

def add_resume_checkpoints(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--resume-skip-task-token",
        action="store_true",
        help="Ignore task token embedding weights in the checkpoint when resuming.",
    )
    parser.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help="Do not load optimizer/scheduler/scaler states when resuming a checkpoint.",
    )
    parser.add_argument(
        "--resume-reset-epoch",
        action="store_true",
        help="Ignore the stored epoch in the checkpoint and restart training from epoch 1.",
    )

def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="VisionARC",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
        help="Optional run name for Weights & Biases.",
    )

def add_speed_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile optimization (useful for debugging).")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode: 'default' (balanced), 'reduce-overhead' (faster), 'max-autotune' (slowest compile, fastest runtime).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision training.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARC-specific Vision Transformer.")
    add_resume_checkpoints(parser)
    add_wandb_args(parser)
    add_speed_optimizer_args(parser)

    parser.add_argument("--data-root", type=str, default="raw_data/ARC-AGI")
    parser.add_argument(
        "--train-split",
        dest="train_split",
        default="training",
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--eval-split",
        dest="eval_split",
        default="training",
        help="Dataset split to evaluate. Use '' to disable evaluation.",
    )
    parser.add_argument(
        "--eval-subset",
        dest="eval_subset",
        choices=("train", "test"),
        default="test",
        help="Which example subset to use when evaluating a split.",
    )
    parser.add_argument("--architecture", type=str, choices=("vit", "unet"), default="vit", help="ViT or UNet architecture")
    parser.add_argument("--unet-size", type=str, choices=("small", "medium", "big"), default="medium", help="Choose the size of UNet wanted; small, medium, or big. Only applies if architecture==unet")
    parser.add_argument("--image-size", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-task-tokens", type=int, default=1, help="Number of task tokens to use in the model.")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=("none", "cosine"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None, help="Optional path to store the final trained checkpoint")
    parser.add_argument("--best-save-path", type=str, default=None, help="Optional path to store the checkpoint achieving the best evaluation accuracy.")

    parser.add_argument("--ttt-num-each", type=int, default=2)
    parser.add_argument("--vis-every", type=int, default=25)
    parser.add_argument("--patch-size", type=int, default=2)

    parser.add_argument(
        "--include-rearc",
        action="store_true",
        help="Add tasks from the Re-ARC dataset to the training set.",
    )
    parser.add_argument(
        "--rearc-path",
        type=str,
        default="raw_data/re_arc",
        help="Path to the Re-ARC dataset root.",
    )
    parser.add_argument(
        "--rearc-limit",
        type=int,
        default=-1,
        help="Maximum number of Re-ARC examples to include per task (use -1 for all examples).",
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed data parallel training (torchrun recommended).",
    )
  
    parser.add_argument('--disable-translation', action='store_true')
    parser.add_argument('--disable-resolution-augmentation', action='store_true')
    parser.add_argument("--fix-scale-factor", type=int, default=1)
    parser.add_argument('--num-attempts', type=int, default=10, help="Number of attempts per evaluation example.")
    parser.add_argument('--eval-save-name', type=str, default=None, help="Name for saving evaluation predictions.")
    
    return parser.parse_args()
