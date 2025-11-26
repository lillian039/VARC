import json
import html
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

IGNORE_INDEX = 10
PAD_INDEX = 11
COLOR_PALETTE = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#B10DC9",
    "#FFFFFF",
]

def _identity_transform(grid):
    return grid


def _ensure_list(grid):
    if grid is None:
        return None
    if isinstance(grid, list):
        return grid
    return grid.tolist()


def _build_task_file_lookup(dataset_root: Optional[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for json_path in dataset_root.glob("*.json"):
        lookup[json_path.stem] = json_path
    return lookup


def _resolve_color_inverse_map(
    task_name: str,
    task_file_lookup: Dict[str, Path],
    cache: Dict[str, Optional[Dict[int, int]]],
) -> Optional[Dict[int, int]]:
    if task_name in cache:
        return cache[task_name]

    task_path = task_file_lookup.get(task_name)
    if task_path is None or not task_path.exists():
        cache[task_name] = None
        return None

    try:
        with task_path.open("r") as fh:
            payload = json.load(fh)
    except Exception:
        cache[task_name] = None
        return None

    color_map = payload.get("augmentation", {}).get("color_map")
    if not color_map:
        cache[task_name] = None
        return None

    normalized = {int(k): int(v) for k, v in color_map.items()}
    inverse_map = {v: k for k, v in normalized.items()}
    cache[task_name] = inverse_map
    return inverse_map


def _apply_color_map_to_grid(grid, inverse_color_map: Optional[Dict[int, int]]):
    if grid is None or not inverse_color_map:
        return grid

    if isinstance(grid, np.ndarray):
        iterable = grid.tolist()
    else:
        iterable = grid

    return [
        [inverse_color_map.get(value, value) for value in row]
        for row in iterable
    ]


def _undo_eval_rot_grid(grid, suffix: str):
    if grid is None or not suffix:
        return grid

    array = np.asarray(grid)
    if array.ndim < 2 or array.size == 0:
        return _ensure_list(array)

    if "rotate_90_" in suffix:
        transformed = np.rot90(array, k=3)
    elif "rotate_180_" in suffix:
        transformed = np.rot90(array, k=2)
    elif "rotate_270_" in suffix:
        transformed = np.rot90(array, k=1)
    elif "flip_0_" in suffix:
        transformed = np.flipud(array)
    elif "flip_1_" in suffix:
        transformed = np.fliplr(array)
    else:
        transformed = array
    return transformed.tolist() if isinstance(transformed, np.ndarray) else transformed


def get_eval_rot_transform_resolver() -> Callable[[str], Tuple[str, Callable]]:
    """Return a resolver that maps eval_rot task names to base ids and undo transforms."""

    def resolver(task_name: str) -> Tuple[str, Callable]:
        if "_" not in task_name:
            return task_name, _identity_transform
        base, suffix = task_name.split("_", 1)

        def undo_fn(grid):
            return _undo_eval_rot_grid(grid, suffix)

        return base, undo_fn

    return resolver


def get_majority_vote(predictions):
    vote_count = {}
    list_map = {}
    for i in range(len(predictions)):
        label = json.dumps(predictions[i])
        list_map[label] = predictions[i]
        if label not in vote_count:
            vote_count[label] = 0
        vote_count[label] += 1
    sorted_votes = sorted(vote_count.items(), key=lambda x: x[1], reverse=True)
    sorted_lists = [
        {"prediction": list_map[item[0]], "votes": item[1]} for item in sorted_votes
    ]
    return sorted_lists

def _grid_to_html_table(grid, title):
    if not grid:
        return f"<div class='grid'><h4>{html.escape(title)}</h4><p class='empty'>No data</p></div>"
    rows = []
    for row in grid:
        cells = []
        for value in row:
            color_index = value if isinstance(value, int) else IGNORE_INDEX
            if not isinstance(value, int):
                try:
                    color_index = int(value)
                except (TypeError, ValueError):
                    color_index = IGNORE_INDEX
            if 0 <= color_index < len(COLOR_PALETTE) - 1:
                color = COLOR_PALETTE[color_index]
            else:
                color = COLOR_PALETTE[-1]
            display_value = "" if color_index == IGNORE_INDEX else str(value)
            cells.append(
                f"<td style='background-color:{color}'>{html.escape(display_value)}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    table = "".join(rows)
    return (
        f"<div class='grid'><h4>{html.escape(title)}</h4>"
        f"<table class='grid-table'>{table}</table></div>"
    )

def one_shot_prediction(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
    save_name = "ttt_eval",
):
    model.eval()
    answer_set = {}

    dataset = getattr(loader, "dataset", None)
    dataset.disable_translation()
    dataset.disable_resolution_augmentation(fix_scale_factor=2)
       
    for batch in tqdm(loader):
        inputs = batch["inputs"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        task_ids = batch["task_ids"].to(device)
        offsets = batch['offset'].to(device)
        scale_factors = batch['scale_factors'].to(device)

        logits = model(inputs, task_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=1).cpu()
        example_indices = batch["example_indices"].cpu()

        for idx, task_name in enumerate(batch["task_names"]):
            # Get rid of augmentation suffixes
            if '_' in task_name:
                continue
            scale_factor = scale_factors[idx].item()
            base_task_name, undo_fn = task_name, _identity_transform

            cur_index = example_indices[idx].item()
            task_predictions = answer_set.setdefault(base_task_name, {})
            if cur_index not in task_predictions:
                task_predictions[cur_index] = []
            
            try:
                offset_x, offset_y = offsets[idx]
                np_predict = np.array(preds[idx]).reshape(img_size, img_size)
                np_predict_grid = np_predict[offset_y:, offset_x:]
                len_x, len_y = 0, 0
                while len_x < np_predict_grid.shape[1] and np_predict_grid[0][len_x] != PAD_INDEX:
                    len_x += 1
                while len_y < np_predict_grid.shape[0] and np_predict_grid[len_y][0] != PAD_INDEX:
                    len_y += 1
                predict_grid = np_predict_grid[:len_y, :len_x].tolist()
                downsampled_grid = []
                for i in range(0, len(predict_grid), scale_factor):
                    row = []
                    for j in range(0, len(predict_grid[0]), scale_factor):
                        block = []
                        for di in range(scale_factor):
                            for dj in range(scale_factor):
                                if i + di < len(predict_grid) and j + dj < len(predict_grid[0]):
                                    block.append(predict_grid[i + di][j + dj])
                        if block:
                            counts = np.bincount(block)
                            majority_value = int(np.argmax(counts))
                            row.append(majority_value)
                    downsampled_grid.append(row)
                predict_grid = downsampled_grid
              
            except Exception as e:
                print("???")
                print(e)
                print(len_y, len_x)
                print(np_predict_grid.shape)
                exit()
                predict_grid = []
            task_predictions[cur_index].append(predict_grid)

    assert len(answer_set.keys()) == 1, "Only support one task for TTT evaluation."
    task_name = list(answer_set.keys())[0]
    os.makedirs(f'outputs/{save_name}', exist_ok=True)
    with open(f'outputs/{save_name}/{task_name}_predictions.json', 'w') as f:
        json.dump(answer_set[task_name], f)
   

    pass
@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
    eval_split: str,
    attempt_nums: int = 10,
    task_transform_resolver: Optional[Callable[[str], Tuple[str, Callable]]] = None,
    border_size: int = 1,
    fix_scale_factor: int = 1,
    disable_translation: bool = False,
    if_fix_scale: bool = False,
    save_name = "ttt_eval",
) -> None:
    model.eval()
    answer_set = {}
    transform_cache: Dict[str, Tuple[str, Callable]] = {}

    dataset = getattr(loader, "dataset", None)
    task_file_lookup: Dict[str, Path] = {}
    color_inverse_cache: Dict[str, Optional[Dict[int, int]]] = {}
    if dataset is not None:
        dataset.enable_translation()
        if disable_translation:
            dataset.disable_translation()
            attempt_nums = 1
        if if_fix_scale:
            dataset.disable_resolution_augmentation(fix_scale_factor=fix_scale_factor)
        else:
            dataset.enable_resolution_augmentation()

        existing_lookup = getattr(dataset, "_task_file_lookup", None)
        if existing_lookup is None:
            dataset_root = getattr(dataset, "root", None)
            dataset_root = Path.joinpath(dataset_root, "data")
            tasks_path = Path.joinpath(dataset_root, eval_split)
            if not tasks_path.exists():
                raise ValueError(f"Tasks path {tasks_path} does not exist.")
            existing_lookup = _build_task_file_lookup(tasks_path)
            setattr(dataset, "_task_file_lookup", existing_lookup)
        task_file_lookup = existing_lookup or {}
    else:
        if disable_translation:
            attempt_nums = 1

    for _ in range(attempt_nums):
        for batch in tqdm(loader):
            inputs = batch["inputs"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            task_ids = batch["task_ids"].to(device)
            offsets = batch['offset'].to(device)
            scale_factors = batch['scale_factors'].to(device)

            logits = model(inputs, task_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=1).cpu()
            example_indices = batch["example_indices"].cpu()

            for idx, task_name in enumerate(batch["task_names"]):
                scale_factor = scale_factors[idx].item()
                if task_transform_resolver:
                    if task_name not in transform_cache:
                        transform_cache[task_name] = task_transform_resolver(task_name)
                    base_task_name, undo_fn = transform_cache[task_name]
                else:
                    base_task_name, undo_fn = task_name, _identity_transform

                cur_index = example_indices[idx].item()
                color_inverse_map = _resolve_color_inverse_map(
                    task_name, task_file_lookup, color_inverse_cache
                )
                print(color_inverse_map)
                task_predictions = answer_set.setdefault(base_task_name, {})
                if cur_index not in task_predictions:
                    task_predictions[cur_index] = []
              
                try:
                    offset_x, offset_y = offsets[idx]
                    np_predict = np.array(preds[idx]).reshape(img_size, img_size)
                    np_predict_grid = np_predict[offset_y:, offset_x:]
                    len_x, len_y = 0, 0
                    while len_x < np_predict_grid.shape[1] and np_predict_grid[0][len_x] != PAD_INDEX:
                        len_x += 1
                    while len_y < np_predict_grid.shape[0] and np_predict_grid[len_y][0] != PAD_INDEX:
                        len_y += 1
                    predict_grid = np_predict_grid[:len_y, :len_x].tolist()
                    predict_grid = undo_fn(predict_grid)
                    if scale_factor > 1:
                        downsampled_grid = []
                        for i in range(0, len(predict_grid), scale_factor):
                            row = []
                            for j in range(0, len(predict_grid[0]), scale_factor):
                                block = []
                                for di in range(scale_factor):
                                    for dj in range(scale_factor):
                                        if i + di < len(predict_grid) and j + dj < len(predict_grid[0]):
                                            block.append(predict_grid[i + di][j + dj])
                                if block:
                                    counts = np.bincount(block)
                                    majority_value = int(np.argmax(counts))
                                    row.append(majority_value)
                            downsampled_grid.append(row)
                        predict_grid = downsampled_grid
                    predict_grid = _apply_color_map_to_grid(
                        predict_grid, color_inverse_map
                    )
                except Exception as e:
                    print("???")
                    print(e)
                    print(len_y, len_x)
                    print(np_predict_grid.shape)
                    exit()
                    predict_grid = []
                task_predictions[cur_index].append(predict_grid)

    assert len(answer_set.keys()) == 1, "Only support one task for TTT evaluation."
    task_name = list(answer_set.keys())[0]
    os.makedirs(f'outputs/{save_name}', exist_ok=True)
    with open(f'outputs/{save_name}/{task_name}_predictions.json', 'w') as f:
        json.dump(answer_set[task_name], f)
   
