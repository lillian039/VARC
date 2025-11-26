"""Utility helpers for augmenting ARC raw data files."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from utils.arclib.arc import Task
from utils.arclib.augmenters import Augmenter, PermuteColors, PermuteColorswithMap
from utils.preprocess import get_augmenters, get_basic_augmenters


def _slugify(label: str) -> str:
    """Generate a filesystem-friendly label from an augmenter description."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug.lower() or "aug"


def _load_task_group(file_path: Path) -> Tuple[Dict[str, object], List[Task]]:
    """Load raw ARC JSON and task variants (one per test example)."""
    with file_path.open("r") as fh:
        data = json.load(fh)

    tasks = Task.read_tasks_from_dict(data)
    if not tasks:
        raise ValueError("No tasks found in ARC file")

    base_name = file_path.stem
    for idx, task in enumerate(tasks):
        if not task.name:
            task.name = f"{base_name}-test{idx}"

    return data, tasks


def _ensure_max_size(task: Task, max_size: int) -> bool:
    return task.max_height() <= max_size and task.max_width() <= max_size


def _make_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed & 0xFFFFFFFF)


def _identity_color_map() -> Dict[int, int]:
    return {i: i for i in range(10)}


def _normalize_color_map(color_map: Dict[int, int]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in color_map.items()}


def _build_payload(
    original_data: Dict[str, object],
    variants: Sequence[Task],
    task_name: str,
) -> Dict[str, object]:
    payload = dict(original_data)
    payload.pop("augmentation", None)

    train_payload = variants[0].serialize()["train"]
    test_payload = [variant.serialize()["test"][0] for variant in variants]

    payload["train"] = train_payload
    payload["test"] = test_payload
    payload["name"] = task_name
    return payload


def _format_filename(template: str, base: str, slug: str, tag: str) -> str:
    """Render a filename template while guarding against missing placeholders."""

    try:
        filename = template.format(base=base, slug=slug, tag=tag)
    except KeyError as exc:  # pragma: no cover - defensive path
        raise ValueError(
            "filename_template must use the placeholders 'base', 'slug', and 'tag'"
        ) from exc

    if not filename:
        raise ValueError("Rendered filename is empty")

    return filename


def augment_raw_data_split_per_task(
    dataset_root: Union[str, Path],
    split: str,
    *,
    only_basic: bool = False,
    output_subdir: Optional[str] = None,
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
    include_random_translate: bool = False,
    augmenters: Optional[Sequence[Augmenter]] = None,
    max_grid_size: int = 30,
    seed: int = 0,
    limit_per_task: Optional[int] = None,
    num_permuate: int = 0,
    augmentation_tag: str = "augmented",
    filename_template: str = "{base}_{slug}_{tag}.json",
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """
    Augment ARC raw data for a given split and persist augmented tasks to disk.

    Args:
        dataset_root: Path to the dataset root (e.g., ``raw_data/ARC-AGI``).
        split: Either ``"training"`` or ``"evaluation"``.
        output_subdir: Optional custom sub-directory name for augmented files. Defaults
            to ``f"{split}_aug"`` under ``dataset_root/data``.
        include_basic/include_size/include_chain/include_repeat/include_concat:
            Flags forwarded to ``get_augmenters`` if ``augmenters`` is not provided.
        augmenters: Explicit list of augmenters to use. When provided the include_* flags
            are ignored.
        max_grid_size: Maximum allowed grid height/width; augmented tasks exceeding
            this are skipped.
        seed: Base seed for deterministic augmentation sampling.
        limit_per_task: Maximum number of augmentations per source task variant. ``None``
            keeps all augmenters.
        num_permuate: Number of additional color permutations (using ``PermuteColors``)
            to generate per augmented task. Set to ``0`` to disable.
        augmentation_tag: String appended to filenames and recorded in metadata
            describing the augmentation family.
        filename_template: Template used for augmented filenames. Must include the
            placeholders ``{base}``, ``{slug}``, and ``{tag}``. The ``slug`` value will
            incorporate permutation indices when applicable.
        dry_run: When ``True`` no files are written; returns the would-be paths.
        verbose: Emit basic progress information.

    Returns:
        List of augmented file paths (actual or prospective if ``dry_run``).
    """
    if split not in {"training", "evaluation"}:
        raise ValueError("split must be 'training' or 'evaluation'")

    dataset_root = Path(dataset_root)
    source_dir = dataset_root / "data" / split
    if not source_dir.exists():
        raise FileNotFoundError(f"Could not find split directory: {source_dir}")

    if output_subdir is None:
        output_dir_root = dataset_root / "data" / f"{split}_aug"
    else:
        output_dir_root = dataset_root / "data" / output_subdir

    if not dry_run:
        output_dir_root.mkdir(parents=True, exist_ok=True)

    all_files = sorted(source_dir.glob("*.json"))
    if not all_files:
        raise RuntimeError(f"No JSON task files found under {source_dir}")

    
    if only_basic:
        augmenters_to_use: Iterable[Augmenter] = get_basic_augmenters()
    elif augmenters is None:
        augmenters_to_use: Iterable[Augmenter] = get_augmenters(
            include_basic=include_basic,
            include_size=include_size,
            include_chain=include_chain,
            include_repeat=include_repeat,
            include_concat=include_concat,
            include_random_translate=include_random_translate
        )
    else:
        augmenters_to_use = augmenters

    augmenters_list = list(augmenters_to_use)
    if verbose:
        print(len(augmenters_list), "augmenters will be applied")
    if not augmenters_list:
        raise ValueError("No augmenters provided")

    saved_paths: List[Path] = []
    total_augments = 0

    for file_index, task_path in enumerate(all_files):
        try:
            original_data, original_tasks = _load_task_group(task_path)
        except Exception as exc:  # pragma: no cover - defensive path
            if verbose:
                print(f"[augment] Failed to load {task_path.name}: {exc}")
            continue
        output_dir = output_dir_root / task_path.stem
        os.makedirs(output_dir, exist_ok=True)

        base_name = task_path.stem
        if not dry_run:
            original_copy_path = output_dir / task_path.name
            if not original_copy_path.exists():
                shutil.copy2(task_path, original_copy_path)
        per_file_count = 0

        for aug_index, augmenter in enumerate(augmenters_list):
            if limit_per_task is not None and per_file_count >= limit_per_task:
                break

            slug = _slugify(str(augmenter))
            filename = _format_filename(
                filename_template,
                base=base_name,
                slug=slug,
                tag=augmentation_tag,
            )
            output_path = output_dir / filename

            rng_seed = (seed * 1_000_003) + (file_index * 97) + aug_index
            rng = _make_rng(rng_seed)
            rng_state = rng.get_state()

            augmented_variants: List[Task] = []
            success = True

            for variant_idx, original_task in enumerate(original_tasks):
                rng.set_state(rng_state)
                try:
                    augmented_task = augmenter.apply_to_task(
                        original_task,
                        rng=rng,
                        share_rng=getattr(augmenter, "share_rng", False),
                        to_input=True,
                        to_output=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    success = False
                    break

                if not _ensure_max_size(augmented_task, max_grid_size):
                    success = False
                    break

                augmented_task.name = f"{base_name}_{slug}"
                augmented_variants.append(augmented_task)

            if not success:
                continue

            base_task_name = augmented_variants[0].name
            payload = _build_payload(original_data, augmented_variants, base_task_name)
            metadata = {
                "augmenter": str(augmenter),
                "source_file": task_path.name,
                "seed": int(rng_seed & 0xFFFFFFFF),
                "test_example_count": len(augmented_variants),
                "color_permutation_index": 0,
                "num_permuate": num_permuate,
                "color_map": _identity_color_map(),
                "augmentation_tag": augmentation_tag,
                "output_filename": output_path.name,
            }
            payload["augmentation"] = metadata

            saved_paths.append(output_path)
            total_augments += 1
            per_file_count += 1

            if not dry_run:
                with output_path.open("w") as fh:
                    json.dump(payload, fh, indent=2)

            if num_permuate <= 0:
                continue

            for perm_idx in range(num_permuate):
                perm_seed = rng_seed + (perm_idx + 1) * 1_048_583
                perm_rng = _make_rng(perm_seed)
                permuted_variants: List[Task] = []
                color_map: Dict[int, int] = {}
                perm_success = True

                for variant_idx, base_variant in enumerate(augmented_variants):
                    if variant_idx == 0:
                        perm_augmenter = PermuteColors()
                        try:
                            permuted_task = perm_augmenter.apply_to_task(
                                base_variant,
                                rng=perm_rng,
                                share_rng=getattr(perm_augmenter, "share_rng", False),
                                to_input=True,
                                to_output=True,
                            )
                        except Exception as exc:  # pragma: no cover - defensive path
                            perm_success = False
                            if verbose:
                                print(
                                    f"[augment] Skipping color permutation {perm_idx + 1} for {task_path.name} with {augmenter}: {exc}"
                                )
                            break
                        color_map = getattr(perm_augmenter, "_color_map", {}) or {}
                    else:
                        perm_mapper = PermuteColorswithMap(color_map)
                        try:
                            permuted_task = perm_mapper.apply_to_task(
                                base_variant,
                                rng=perm_rng,
                                share_rng=getattr(perm_mapper, "share_rng", False),
                                to_input=True,
                                to_output=True,
                            )
                        except Exception as exc:  # pragma: no cover - defensive path
                            perm_success = False
                            if verbose:
                                print(
                                    f"[augment] Skipping color permutation {perm_idx + 1} for {task_path.name} with {augmenter} (test #{variant_idx}): {exc}"
                                )
                            break
                        

                    permuted_task.name = f"{base_name}_{slug}_perm{perm_idx + 1}"
                    permuted_variants.append(permuted_task)
                    

                if not perm_success:
                    continue

                color_map_normalized = _normalize_color_map(color_map)
                perm_payload = _build_payload(
                    original_data,
                    permuted_variants,
                    permuted_variants[0].name,
                )
                perm_metadata = {
                    "augmenter": str(augmenter),
                    "source_file": task_path.name,
                    "seed": int(perm_seed & 0xFFFFFFFF),
                    "test_example_count": len(permuted_variants),
                    "color_permutation_index": perm_idx + 1,
                    "num_permuate": num_permuate,
                    "color_map": color_map_normalized,
                    "augmentation_tag": augmentation_tag,
                }
                perm_payload["augmentation"] = perm_metadata

                perm_slug = f"{slug}_perm{perm_idx + 1}"
                perm_filename = _format_filename(
                    filename_template,
                    base=base_name,
                    slug=perm_slug,
                    tag=augmentation_tag,
                )
                perm_output_path = output_dir / perm_filename

                perm_metadata["output_filename"] = perm_output_path.name

                saved_paths.append(perm_output_path)

                total_augments += 1
                

                if not dry_run:
                    with perm_output_path.open("w") as fh:
                        json.dump(perm_payload, fh, indent=2)

    if verbose:
        print(
            f"[augment] Completed split '{split}': {total_augments} augmented tasks prepared in {output_dir}"
        )

    return saved_paths
