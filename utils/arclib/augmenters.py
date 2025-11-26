# https://github.com/ekinakyurek/marc
"""
Augmenters for ARC tasks

The augmenters are used to apply transformations to the input and output of the tasks.

Augmenter: Base class for all augmenters
Rotate: Rotate the grid by 90, 180, 270 degrees
PermuteColors: Permute the colors in the grid
PermuteColorswithMap: Permute the colors in the grid using a given color map
PermuteColorsRespectKeyColors: Permute the colors in the grid while keeping the key colors fixed
Flip: Flip the grid along the given axis
Reflect: Reflect the grid along the given axis
Repeat: Repeat the grid along the given
Transpose: Transpose the grid
IncreaseResolution: Increase the resolution of the grid
IncreaseHeight: Increase the height of the grid
IncreaseWidth: Increase the width of the grid
DropoutInput: Delete a random rectangular patch from the input
DropoutOutput: Delete a random rectangular patch from the output
RandomTranslateXY: Randomly shift the grid along the x and y axis
RandomTranslateX: Randomly shift the grid along the x axis
RandomTranslateY: Randomly shift the grid along the y axis
Chain: Chain multiple augmenters together
Concat: Concatenate the output of multiple augmenters
IdentityAugmenter: Identity augmenter
PermuteExamples: Permute the training examples in the task
"""
import copy
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import RandomState

from .arc import Example, Grid, Task


class Augmenter:
    share_rng: bool = False

    def __repr__(self):
        return str(self)

    def __call__(self, grid: Grid, rng: RandomState = None) -> Grid:
        return self.apply_to_grid(grid, rng=rng)

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        raise NotImplementedError()

    def apply_to_example(
        self,
        example: Example,
        rng: RandomState = None,
        to_input: bool = True,
        to_output: bool = True,
    ) -> Example:
        input = self.apply_to_grid(example.input, rng=rng) if to_input else example.input
        output = self.apply_to_grid(example.output, rng=rng) if to_output else example.output
        if example.cot is not None:
            cot = copy.deepcopy(example.cot)
            if not np.array_equal(cot[-1], output):
                cot.append(output)
        else:
            cot = None
        return Example(input, output, cot)

    def apply_to_task(self, task: Task, rng: RandomState = None, share_rng=False, **kwargs) -> Task:

        if self.share_rng:
            # make sure all examples get seperate copies of same rng
            # this is to make sure that the same random number(s) is used for all examples
            # seed = rng.randint(0, 2**32)
            train_rngs = [copy.deepcopy(rng) for i in range(len(task.train_examples))]
            test_rng = copy.deepcopy(rng)  # RandomState(seed)
        else:
            train_rngs = [rng for _ in range(len(task.train_examples))]
            test_rng = rng

        return Task(
            train_examples=[
                self.apply_to_example(example, rng=rng_i, **kwargs)
                for example, rng_i in zip(task.train_examples, train_rngs)
            ],
            test_example=self.apply_to_example(task.test_example, rng=test_rng, **kwargs),
            name=task.name,
        )


class Rotate(Augmenter):
    def __init__(self, angle: int):
        assert angle in [0, 90, 180, 270]
        self.angle = angle

    def __str__(self):
        return f"Rotate({self.angle})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        # roate the input and output by the given angle
        del rng
        if self.angle == 90:
            return np.rot90(grid, k=1)
        elif self.angle == 180:
            return np.rot90(grid, k=2)
        elif self.angle == 270:
            return np.rot90(grid, k=3)
        else:
            raise ValueError("Invalid angle")


class PermuteColors(Augmenter):
    share_rng = True
    color_mapper = None

    def __str__(self):
        return "PermuteColors()"

    def apply_to_task(
        self,
        task: Task,
        use_test_output: bool = True,
        rng: RandomState = None,
        share_rng=False,
        **kwargs,
    ) -> Task:
        # get all used colors in the inputs and outputs
        colors = []
        for example in task.train_examples:
            colors += example.input.flatten().tolist()
            colors += example.output.flatten().tolist()
        colors += task.test_example.input.flatten().tolist()

        if use_test_output:
            colors += task.test_example.output.flatten().tolist()

        # get unique colors
        colors = set(colors)
        # remove 0
        colors = colors - {0}
        remaining_colors = list(set(list(range(1, 10))) - colors)
        colors = list(colors)

        rng.shuffle(remaining_colors)  # inplac

        permuted_colors = rng.permutation(colors).tolist()
        # sample a mapping from colors to new ids
        color_map = {0: 0}

        for color in colors:

            if color in color_map:
                continue

            if len(remaining_colors) > 0:
                new_color = remaining_colors.pop()
            else:
                new_color = permuted_colors.pop()
                # in this case we want to directly swap colors
                # so new_color shoud be the color that is being replaced
                color_map[new_color] = color
                # remove color from permuted colors
                if color in permuted_colors:
                    permuted_colors.remove(color)

            color_map[color] = new_color

        self._color_map = color_map

        def color_mapper(color: int) -> int:
            return color_map.get(color, color)

        self.color_mapper = np.vectorize(color_mapper)

        return Task(
            train_examples=[
                self.apply_to_example(example, rng=rng, **kwargs) for example in task.train_examples
            ],
            test_example=self.apply_to_example(task.test_example, rng=rng, **kwargs),
            name=task.name,
        )

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        # map colors to new colors
        return self.color_mapper(grid)


class PermuteColorswithMap(Augmenter):

    def __init__(self, color_map):
        self.color_map = color_map

        def color_mapper(color: int) -> int:
            return color_map.get(color, color)

        self.color_mapper = np.vectorize(color_mapper)

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        # map colors to new colors
        return self.color_mapper(grid)

    def __repr__(self):
        return f"PermuteColorswithMap({self.color_map})"

    def __str__(self):
        return self.__repr__()


class PermuteColorsRespectKeyColors(Augmenter):
    share_rng = True
    color_mapper = None

    def __init__(
        self,
        key_colors: List[int] = None,
        use_remaining_colors: bool = True,
        use_test_output: bool = False,
    ):
        self.key_colors = key_colors
        self.use_remaining_colors = use_remaining_colors
        self.use_test_output = use_test_output

    def __str__(self):
        return "PermuteColorsRespectKeyColors(key_colors={self.key_colors}, use_remaining_colors={self.use_remaining_colors}, use_test_output={self.use_test_output})"

    @staticmethod
    def get_key_colors(task: Task, use_test_output: bool = False):
        key_colors_input = []
        key_colors_output = []
        for example in task.train_examples:
            key_colors_input.append(set(example.input.flatten().tolist()))
            key_colors_output.append(set(example.output.flatten().tolist()))

        key_colors_input.append(set(task.test_example.input.flatten().tolist()))
        if use_test_output:
            key_colors_output.append(set(task.test_example.output.flatten().tolist()))

        all_colors = set.union(*key_colors_input, *key_colors_output)

        key_colors_input = set.intersection(*key_colors_input)
        key_colors_output = set.intersection(*key_colors_output)
        key_colors = key_colors_input.union(key_colors_output)

        return key_colors, all_colors

    def apply_to_task(self, task: Task, rng: RandomState = None, share_rng=False, **kwargs) -> Task:
        key_colors, colors = PermuteColorsRespectKeyColors.get_key_colors(
            task, use_test_output=self.use_test_output
        )
        if self.key_colors is None:
            self.key_colors = key_colors
        else:
            key_colors = self.key_colors

        remaining_colors = list(set(list(range(10))) - colors)
        colors = list(colors)

        rng.shuffle(remaining_colors)  # inplace
        # keep key colors_static
        color_map = {}

        for key_color in self.key_colors:
            color_map[key_color] = key_color
            if key_color in colors:
                colors.remove(key_color)

        permuted_colors = rng.permutation(colors).tolist()

        for color in colors:

            if color in color_map:
                continue

            if self.use_remaining_colors and len(remaining_colors) > 0:
                new_color = remaining_colors.pop()
            else:
                new_color = permuted_colors.pop()
                # in this case we want to directly swap colors
                # so new_color shoud be the color that is being replaced
                color_map[new_color] = color
                # remove color from permuted colors
                if color in permuted_colors:
                    permuted_colors.remove(color)

            color_map[color] = new_color

        def color_mapper(color: int) -> int:
            return color_map.get(color, color)

        self.color_mapper = np.vectorize(color_mapper)

        return Task(
            train_examples=[
                self.apply_to_example(example, rng=rng, **kwargs) for example in task.train_examples
            ],
            test_example=self.apply_to_example(task.test_example, rng=rng, **kwargs),
            name=task.name,
        )

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        # map colors to new colors
        return self.color_mapper(grid)


class PermuteColorsStatic(Augmenter):
    share_rng = True
    color_mapper = None

    def __str__(self):
        return f"PermuteColorsStatic({self.color_map})"

    def __init__(self, color_map):
        self.color_map = color_map

        self.color_mapper = np.vectorize(lambda x: self.color_map.get(x, x))

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        return self.color_mapper(grid)


class Flip(Augmenter):
    def __init__(self, axis: int):
        assert axis in [0, 1]
        self.axis = axis

    def __str__(self):
        return f"Flip({self.axis})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        del rng
        if self.axis == 0:
            return np.flipud(grid)
        elif self.axis == 1:
            return np.fliplr(grid)
        else:
            raise ValueError("Invalid axis")


class Reflect(Augmenter):
    def __init__(self, axis: int, reverse=False):
        assert axis in [0, 1]
        self.axis = axis
        self.reverse = reverse

    def __str__(self):
        return f"Reflect({self.axis}, reverse={self.reverse})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Example:
        del rng
        # reflect the input and output by the given axis
        original = grid
        if self.axis == 0:
            reflected = np.flipud(original)
        elif self.axis == 1:
            reflected = np.fliplr(original)
        else:
            raise ValueError("Invalid axis")

        if self.reverse:
            if self.axis == 0:
                return np.concatenate((reflected, original), axis=0)
            elif self.axis == 1:
                return np.concatenate((reflected, original), axis=1)
        else:
            if self.axis == 0:
                return np.concatenate((original, reflected), axis=0)
            elif self.axis == 1:
                return np.concatenate((original, reflected), axis=1)


class Repeat(Augmenter):
    def __init__(self, axis: int, n: int):
        assert axis in [0, 1, 2]
        self.axis = axis
        self.n = n

    def __str__(self):
        return f"Repeat({self.axis}, {self.n})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        del rng
        if self.axis == 0:
            return np.concatenate([grid] * self.n, axis=0)
        elif self.axis == 1:
            return np.concatenate([grid] * self.n, axis=1)
        elif self.axis == 2:
            return np.concatenate([np.concatenate([grid] * self.n, axis=0)] * self.n, axis=1)
        else:
            raise ValueError("Invalid axis")


class Transpose(Augmenter):
    def __str__(self):
        return "Transpose()"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        del rng
        return np.transpose(grid)


class IncreaseResolution(Augmenter):
    def __init__(self, factor: int):
        assert factor > 1
        self.factor = factor

    def __str__(self):
        return f"IncreaseResolution({self.factor})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        del rng
        grid = np.repeat(grid, self.factor, axis=0)
        grid = np.repeat(grid, self.factor, axis=1)
        return grid


class IncreaseHeight(Augmenter):
    def __init__(self, factor: int):
        assert factor > 1
        self.factor = factor

    def __str__(self):
        return f"IncreaseHeight({self.factor})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        del rng
        grid = np.repeat(grid, self.factor, axis=0)
        return grid


class IncreaseWidth(Augmenter):
    def __init__(self, factor: int):
        assert factor > 1
        self.factor = factor

    def __str__(self):
        return f"IncreaseWidth({self.factor})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        del rng
        grid = np.repeat(grid, self.factor, axis=1)
        return grid


class DropoutInput(Augmenter):
    """
    Delete a random rectangular patch
    """

    def __init__(self):
        self.dropout_color = 0

    def __str__(self):
        return "DropoutInput()"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        # get rng with seed
        if rng is None:
            rng = RandomState()
        grid = grid.copy()
        num_dropouts = rng.randint(1, 3)
        for _ in range(num_dropouts):
            x_len_ratio = rng.uniform(0.1, 0.3)
            y_len_ratio = rng.uniform(0.1, 0.3)
            x_start_ratio = rng.uniform(0.1, 0.7)
            y_start_ratio = rng.uniform(0.1, 0.7)

            x_len = int(np.ceil(grid.shape[0] * x_len_ratio))
            y_len = int(np.ceil(grid.shape[1] * y_len_ratio))

            x_start = int(grid.shape[0] * x_start_ratio)
            y_start = int(grid.shape[1] * y_start_ratio)

            grid[x_start : x_start + x_len, y_start : y_start + y_len] = self.dropout_color
        return grid

    def apply_to_task(self, task: Task, rng: RandomState = None, share_rng=False, **kwargs) -> Task:
        # find unused colors in the task
        all_colors = set()
        for example in task.train_examples:
            all_colors.update(set(example.input.flatten().tolist()))
            all_colors.update(set(example.output.flatten().tolist()))

        all_colors = list(set(range(10)) - all_colors)
        if len(all_colors) == 0:
            self.dropout_color = 0
        else:
            self.dropout_color = rng.choice(all_colors)

        return super().apply_to_task(task, rng=rng, share_rng=share_rng, **kwargs)


class DropoutOutput(Augmenter):
    share_rng = True

    def __str__(self):
        return "DropoutOutput()"
        self.dropout_color = 0

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        assert rng is not None
        grid = grid.copy()
        num_dropouts = rng.randint(1, 3)
        for _ in range(num_dropouts):
            x_len_ratio = rng.uniform(0.1, 0.3)
            y_len_ratio = rng.uniform(0.1, 0.3)
            x_start_ratio = rng.uniform(0.1, 0.7)
            y_start_ratio = rng.uniform(0.1, 0.7)

            x_len = int(np.ceil(grid.shape[0] * x_len_ratio))
            y_len = int(np.ceil(grid.shape[1] * y_len_ratio))

            x_start = int(grid.shape[0] * x_start_ratio)
            y_start = int(grid.shape[1] * y_start_ratio)

            grid[x_start : x_start + x_len, y_start : y_start + y_len] = self.dropout_color
        return grid

    def apply_to_task(self, task: Task, rng: RandomState = None, share_rng=False, **kwargs) -> Task:
        # find unused colors in the task
        all_colors = set()
        for example in task.train_examples:
            all_colors.update(set(example.input.flatten().tolist()))
            all_colors.update(set(example.output.flatten().tolist()))

        all_colors = list(set(range(10)) - all_colors)

        if len(all_colors) == 0:
            self.dropout_color = 0
        else:
            self.dropout_color = rng.choice(all_colors)

        return super().apply_to_task(task, rng=rng, share_rng=share_rng, **kwargs)


class RandomTranslateXY(Augmenter):
    share_rng = True

    def __str__(self):
        return "RandomTranslateXY()"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        assert rng is not None
        # if rng.rand() < 0.5:
        shift_x = rng.randint(0, min(4, grid.shape[0]))
        shift_y = rng.randint(0, min(4, grid.shape[1]))
        # else:
        #     shift_x_ratio = rng.uniform(-0.5, 0.5)
        #     shift_x = int(np.round(grid.shape[0] * shift_x_ratio))
        #     shift_y_ratio = rng.uniform(-0.5, 0.5)
        #     shift_y = int(np.round(grid.shape[1] * shift_y_ratio))
        return np.roll(grid, (shift_x, shift_y), axis=(0, 1))


class RandomTranslateX(Augmenter):
    share_rng = True

    def __str__(self):
        return "RandomTranslateX()"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        assert rng is not None
        # if rng.rand() < 0.5:
        shift_x = rng.randint(1, min(4, grid.shape[0]))
        # else:
        #     shift_x_ratio = rng.uniform(-0.5, 0.5)
        #     shift_x = int(np.round(grid.shape[0] * shift_x_ratio))
        return np.roll(grid, shift_x, axis=0)


class RandomTranslateY(Augmenter):
    share_rng = True

    def __str__(self):
        return "RandomTranslateY()"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        assert rng is not None
        # if rng.rand() < 0.5:
        shift_y = rng.randint(1, min(4, grid.shape[1]))
        # else:
        #     shift_y_ratio = rng.uniform(-0.5, 0.5)
        #     shift_y = int(np.round(grid.shape[1] * shift_y_ratio))
        return np.roll(grid, shift_y, axis=1)


def find_connected_components(
    grid: Grid, background_color: Optional[int] = None
) -> List[List[tuple[int, int]]]:
    # get background color as the most frequenct color
    if background_color is None:
        background_color = np.bincount(grid.flatten()).argmax()

    # get connected components
    visited = np.zeros_like(grid)
    components = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if visited[i, j] == 0 and grid[i, j] != background_color:
                component = []
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
                        continue
                    if visited[x, y] == 1 or grid[x, y] == background_color:
                        continue
                    component.append((x, y))
                    visited[x, y] = 1
                    stack.append((x + 1, y))
                    stack.append((x - 1, y))
                    stack.append((x, y + 1))
                    stack.append((x, y - 1))
                components.append(component)

    return components, background_color


class RandomObjectRotate(Augmenter):
    share_rng = True

    def __init__(self, angle: int):
        self.angle = angle

    def __str__(self):
        return f"RandomObjectRotate({self.angle})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None, background_color: int = 0) -> Grid:
        assert rng is not None

        components, background_color = find_connected_components(
            grid, background_color=background_color
        )

        if not components:
            return grid

        idx = rng.choice(len(components), size=1)[0]
        component = components[idx]

        # copy grid
        grid = grid.copy()

        # rotate part of the grid by the given angle assume left bottom corner is the origin
        rotated_component = []
        origin = np.array([min(x for x, y in component), min(y for x, y in component)])
        for x, y in component:
            color = grid[x, y]
            x -= origin[0]
            y -= origin[1]
            if self.angle == 90:
                x, y = y, -x
            elif self.angle == 180:
                x, y = -x, -y
            elif self.angle == 270:
                x, y = -y, x
            rotated_component.append((x + origin[0], y + origin[1], color))

        for x, y in component:
            grid[x, y] = background_color
        for x, y, color in rotated_component:
            grid[x, y] = color

        return grid


class PermuteExamples(Augmenter):
    def apply_to_task(self, task: Task, rng: RandomState = None, share_rng=False, **kwargs) -> Task:
        if rng is None:
            rng = RandomState()

        perm = rng.permutation(len(task.train_examples))
        train_examples = [task.train_examples[i] for i in perm]
        return Task(train_examples=train_examples, test_example=task.test_example, name=task.name)


class RandomObjectTranslateXY(Augmenter):
    share_rng = True

    def __str__(self):
        return "RandomObjectTranslateXY()"

    def apply_to_grid(
        self, grid: Grid, rng: RandomState = None, background_color: Optional[int] = None
    ) -> Grid:
        assert rng is not None

        components, background_color = find_connected_components(
            grid, background_color=background_color
        )

        if not components:
            return grid

        idx = rng.choice(len(components), size=1)[0]
        component = components[idx]

        # copy grid
        grid = grid.copy()

        # translate part of the grid by the given angle assume left bottom corner is the origin
        translated_component = []
        shift_x = rng.randint(-grid.shape[0] // 2, grid.shape[0] // 2)
        shift_y = rng.randint(-grid.shape[1] // 2, grid.shape[1] // 2)
        for x, y in component:
            color = grid[x, y]
            x += shift_x
            y += shift_y
            translated_component.append((x, y, color))

        for x, y in component:
            grid[x, y] = background_color
        for x, y, color in translated_component:
            grid[x, y] = color

        return grid


class Chain(Augmenter):
    def __init__(self, augmenters: Tuple[Augmenter]):
        self.augmenters = augmenters

    def __str__(self):
        return f"Chain({self.augmenters})"

    def apply_to_task(self, task: Task, rng: RandomState = None, share_rng=False, **kwargs) -> Task:
        for augmenter in self.augmenters:
            task = augmenter.apply_to_task(task, rng=rng, share_rng=augmenter.share_rng, **kwargs)
        return task

    def apply_to_example(
        self,
        example: Example,
        rng: RandomState = None,
        to_input: bool = True,
        to_output: bool = True,
    ) -> Example:
        for augmenter in self.augmenters:
            example = augmenter.apply_to_example(
                example, rng=rng, to_input=to_input, to_output=to_output
            )
        return example

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        for augmenter in self.augmenters:
            grid = augmenter.apply_to_grid(grid, rng=rng)
        return grid


class Concat(Augmenter):
    def __init__(self, augmenters: Tuple[Augmenter], axis: int = 0):
        self.augmenters = augmenters
        self.axis = axis

    def __str__(self):
        return f"Concat({self.augmenters}, axis={self.axis})"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        grids = []
        for augmenter in self.augmenters:
            grids.append(augmenter.apply_to_grid(grid, rng=rng))

        return np.concatenate(grids, axis=self.axis)


class IdentityAugmenter(Augmenter):
    def __str__(self):
        return "IdentityAugmenter()"

    def apply_to_grid(self, grid: Grid, rng: RandomState = None) -> Grid:
        return grid.copy()


def inverse(augmenter):
    if isinstance(augmenter, Rotate):
        return Rotate(360 - augmenter.angle)
    elif isinstance(augmenter, Flip):
        return augmenter
    elif isinstance(augmenter, Transpose):
        return augmenter
    elif isinstance(augmenter, PermuteColors):
        color_map = augmenter._color_map
        # reverse
        inverse_map = {v: k for k, v in color_map.items()}
        return PermuteColorswithMap(inverse_map)


augmenters_to_apply_to_input = [
    DropoutInput(),
    IncreaseResolution(2),
    IncreaseResolution(3),
    IncreaseHeight(2),
    IncreaseWidth(2),
    IncreaseHeight(3),
    IncreaseWidth(3),
]

input_augmenters_probs = [
    1,
    1 / 2,
    1 / 2,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
]

augmenters_to_apply_to_output = [
    Rotate(90),
    Rotate(270),
    Rotate(180),
    Flip(0),
    Flip(1),
    Reflect(0, reverse=True),
    Reflect(1, reverse=True),
    Reflect(0, reverse=False),
    Reflect(1, reverse=False),
    IncreaseResolution(2),
    IncreaseResolution(3),
    IncreaseHeight(2),
    IncreaseWidth(2),
    Transpose(),
    RandomTranslateXY(),
    DropoutOutput(),
    Repeat(0, 2),
    Repeat(1, 2),
    Repeat(0, 3),
    Repeat(1, 3),
    Repeat(2, 2),
    Repeat(2, 3),
]

output_augmenters_probs = [
    1 / 3,
    1 / 3,
    1 / 3,
    1 / 2,
    1 / 2,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 2,
    1 / 2,
    1 / 2,
    1 / 5,
    1 / 5,
    1 / 5,
    1 / 5,
    1 / 5,
    1 / 5,
]

augmenters_to_apply_to_both = [
    IncreaseResolution(2),
    IncreaseResolution(3),
    IncreaseHeight(2),
    IncreaseWidth(2),
    IncreaseHeight(3),
    IncreaseWidth(3),
    PermuteColors(),
    # Transpose(),
    # Rotate(90),
    # Rotate(270),
    # Rotate(180),
    # Flip(0),
    # Flip(1),
    # Reflect(0, reverse=False),
    # Reflect(1, reverse=False),
]

both_augmenters_probs = [
    1 / 2,
    1 / 2,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 2,
    # 1/8,
    # 1/8,
    # 1/8,
    # 1/8,
    # 1/8,
    # 1/8,
    # 1/8,
    # 1/8,
]

# normalize
input_augmenters_probs = [p / sum(input_augmenters_probs) for p in input_augmenters_probs]
output_augmenters_probs = [p / sum(output_augmenters_probs) for p in output_augmenters_probs]
both_augmenters_probs = [p / sum(both_augmenters_probs) for p in both_augmenters_probs]


def apply_a_random_augmentation(task: Task, rng=None) -> Tuple[Task, Augmenter, str]:
    if rng is None:
        rng = RandomState()

    category = rng.choice(["input", "output", "both"], p=[0.3, 0.6, 0.1])

    if category == "input":
        augmenter = rng.choice(augmenters_to_apply_to_input, p=input_augmenters_probs)
        task = augmenter.apply_to_task(task, to_input=True, to_output=False, rng=rng)
    elif category == "output":
        augmenter = rng.choice(augmenters_to_apply_to_output, p=output_augmenters_probs)
        task = augmenter.apply_to_task(task, to_input=False, to_output=True, rng=rng)
    else:
        augmenter = rng.choice(augmenters_to_apply_to_both, p=both_augmenters_probs)
        task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)

    return task, augmenter, category


if __name__ == "__main__":

    grid = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5],
        ]
    )

    rng = RandomState(45)
    task = Task(train_examples=[Example(grid, grid)], test_example=Example(grid, grid))

    drop_task = DropoutOutput().apply_to_task(task, rng=rng, to_input=False, to_output=True)
    drop_task_input = DropoutInput().apply_to_task(task, rng=rng, to_input=True, to_output=False)

    grid1 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 2, 2, 0, 0],
        ]
    )

    # another grid
    grid2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
            [0, 2, 4, 0, 0],
            [0, 9, 4, 0, 0],
            [0, 9, 4, 0, 0],
        ]
    )

    train_examples = [Example(grid1, grid1) for _ in range(3)]
    train_examples.append(Example(grid1, grid2))
    test_example = Example(grid1, grid1)

    task = Task(train_examples=train_examples, test_example=test_example)

    for _ in range(1000):
        ttask, augmenterr, category = apply_a_random_augmentation(task, rng=rng)
        assert np.array_equal(ttask.train_examples[0].output, ttask.train_examples[1].output)
        assert np.array_equal(ttask.train_examples[0].output, ttask.train_examples[2].output)
        assert not np.array_equal(ttask.train_examples[0].output, ttask.train_examples[3].output)

    # apply permute colors
    permute_colors = PermuteColors()
    ttask = permute_colors.apply_to_task(task, rng=rng)

    # random permutation
    permuted = rng.permutation(np.arange(10))

    print(permuted)

    def color_mapper(color: int) -> int:
        return permuted[color]

    color_mapper_v = np.vectorize(color_mapper)

    color_mapper_v(task.test_example.input)
