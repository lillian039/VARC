import numpy as np
from PIL import Image

COLORS_UINT8 = {
    0: [0, 0, 0],
    1: [0, 116, 217],
    2: [255, 65, 54],
    3: [46, 204, 64],
    4: [255, 220, 0],
    5: [170, 170, 170],
    6: [240, 18, 190],
    7: [255, 133, 27],
    8: [127, 219, 255],
    9: [135, 12, 37],
    10: [200, 200, 200],
    11: [255, 255, 255],
}
COLORS_UINT8 = np.array([COLORS_UINT8[i] for i in range(len(COLORS_UINT8.keys()))], dtype=np.uint8)
PIL_IMG_MAG = 10
BORDER_SIZE = 4

def _to_numpy(tensor_like):
    if tensor_like is None:
        return None
    if isinstance(tensor_like, np.ndarray):
        return tensor_like
    if hasattr(tensor_like, "detach"):
        tensor_like = tensor_like.detach()
    if hasattr(tensor_like, "cpu"):
        tensor_like = tensor_like.cpu()
    if hasattr(tensor_like, "numpy"):
        tensor_like = tensor_like.numpy()
    return np.array(tensor_like)


def _compute_crop(mask, *extra_masks):
    mask_np = _to_numpy(mask)
    if mask_np is None:
        raise ValueError("Primary mask must be provided for visualization.")
    mask_np = mask_np.astype(bool)

    combined = mask_np
    for extra in extra_masks:
        extra_np = _to_numpy(extra)
        if extra_np is None:
            continue
        combined = np.logical_or(combined, extra_np.astype(bool))

    valid_coords = np.argwhere(combined)
    if valid_coords.size == 0:
        height, width = combined.shape
        return slice(0, height), slice(0, width)

    min_row, min_col = valid_coords.min(axis=0)
    max_row, max_col = valid_coords.max(axis=0) + 1
    return slice(min_row, max_row), slice(min_col, max_col)


def grid_to_pil(mask, input_grid=None, target_grid=None, pred_grid=None, IGNORE_INDEX=-100):
    target_mask = None
    if target_grid is not None:
        target_mask = target_grid != IGNORE_INDEX

    row_slice, col_slice = _compute_crop(mask, target_mask)

    if input_grid is not None:
        input_grid = input_grid[row_slice, col_slice]
    if target_grid is not None:
        target_grid = target_grid[row_slice, col_slice]
    if pred_grid is not None:
        pred_grid = pred_grid[row_slice, col_slice]

    max_h, max_w = 32, 32

    img = np.full(((max_h+8)*PIL_IMG_MAG, 3*max_w*PIL_IMG_MAG+2*BORDER_SIZE, 4), 0, dtype=np.uint8)
    for idx, grid in enumerate([input_grid, target_grid, pred_grid]):
        if grid is None:
            continue
        grid = _to_numpy(grid)
        grid = grid.astype(int, copy=False)
        mult_factor = min((max_h*PIL_IMG_MAG)//max(grid.shape[0], 1), (max_w*PIL_IMG_MAG)//max(grid.shape[1], 1))
        mult_factor = max(mult_factor, 1)

        grid = np.repeat(np.repeat(grid, mult_factor, axis=0), mult_factor, axis=1)
        safe_indices = np.where((grid < 0) | (grid >= COLORS_UINT8.shape[0]), 0, grid)
        grid = COLORS_UINT8[safe_indices, :]
        grid[::mult_factor, :, :] = 100
        grid[mult_factor-1::mult_factor, :, :] = 100
        grid[:, ::mult_factor, :] = 100
        grid[:, mult_factor-1::mult_factor, :] = 100

        start_h = ((max_h+4) * PIL_IMG_MAG - grid.shape[0])//2
        start_w = idx * (max_w * PIL_IMG_MAG + BORDER_SIZE) + (max_w * PIL_IMG_MAG - grid.shape[1])//2
        img[start_h:start_h + grid.shape[0], start_w:start_w + grid.shape[1], :3] = grid
        img[start_h:start_h + grid.shape[0], start_w:start_w + grid.shape[1], -1] = 255

    return Image.fromarray(img, mode="RGBA")


def grid_to_pil_all(input_grid=None, target_grid=None, grid_line_color=(255, 255, 255)):
    max_h, max_w = 32, 32

    images = []
    line_color = np.array(grid_line_color, dtype=np.uint8)
    if line_color.ndim == 0:
        line_color = np.full(3, line_color, dtype=np.uint8)

    for grid in (input_grid, target_grid):
        if grid is None:
            images.append(None)
            continue
        grid = _to_numpy(grid)
        grid = grid.astype(int, copy=False)
        mult_factor = min(
            (max_h * PIL_IMG_MAG) // max(grid.shape[0], 1),
            (max_w * PIL_IMG_MAG) // max(grid.shape[1], 1),
        )
        mult_factor = max(mult_factor, 1)

        grid = np.repeat(np.repeat(grid, mult_factor, axis=0), mult_factor, axis=1)
        safe_indices = np.where((grid < 0) | (grid >= COLORS_UINT8.shape[0]), 0, grid)
        color_grid = COLORS_UINT8[safe_indices, :].copy()
        color_grid[::mult_factor, :, :] = line_color
        color_grid[mult_factor - 1 :: mult_factor, :, :] = line_color
        color_grid[:, ::mult_factor, :] = line_color
        color_grid[:, mult_factor - 1 :: mult_factor, :] = line_color

        h, w, _ = color_grid.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = color_grid
        rgba[:, :, 3] = 255
        images.append(Image.fromarray(rgba, mode="RGBA"))

    return images[0], images[1]
