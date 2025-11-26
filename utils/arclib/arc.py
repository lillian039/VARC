# https://github.com/ekinakyurek/marc
"""
This module contains classes to represent ARC tasks and examples

Grid: a numpy array representing a grid
Example: a class to represent an example (example.input and example.output are grids)
Task: a class to represent a task (task.test_example and task.train_examples are test and train examples)
read_from_single_file: a function to read challenge problems and solutions from a single file
make_submission: a function to create a submission file
"""
import dataclasses
import glob
import json
import os
from typing import List, Optional

import numpy as np


Grid = np.ndarray


def to_tuple(arr):
    return tuple(tuple([int(e) for e in row]) for row in arr)


def to_list(arr):
    return [[int(e) for e in row] for row in arr]


@dataclasses.dataclass
class Example:
    """
    class to represent an example
    """

    input: Grid
    output: Grid
    cot: Optional[List[Grid]] = None

    def input_size(self) -> int:
        """return the size of the input grid"""
        return self.input.size

    def output_size(self) -> int:
        """return the size of the output grid"""
        return self.output.size

    def size(self) -> int:
        """return the size of the example"""
        return max(self.input_size(), self.output_size())

    def __hash__(self) -> int:
        return hash((self.input.tobytes(), self.output.tobytes()))

    def __repr__(self) -> str:
        return f"Example(input={self.input}, output={self.output})"

    def serialize(self) -> dict:
        example = {"input": self.input.tolist(), "output": self.output.tolist()}

        if self.cot:
            example["cot"] = [cot.tolist() for cot in self.cot]

        return example

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Example):
            return NotImplemented
        return np.array_equal(self.input, other.input) and np.array_equal(self.output, other.output)

    @classmethod
    def deserialize(cls, data: dict, test: bool = False) -> "Example":
        input = np.array(data["input"])
        if test:
            output = input.copy()
        elif "output" in data:
            output = np.array(data["output"])
        else:
            output = input.copy()
        cot = None
        if "cot" in data:
            cot = [np.array(c) for c in data["cot"]]
        return cls(input, output, cot)


@dataclasses.dataclass
class Task:
    """
    A class to represent a task
    """

    test_example: Example
    train_examples: List[Example] = dataclasses.field(default_factory=list)
    name: str = ""

    def size(self) -> int:
        """return the size of the task"""
        return max([example.size() for example in self.train_examples])

    def max_height(self) -> int:
        max_x = 0
        for example in self.train_examples:
            x, _ = example.input.shape
            max_x = max(max_x, x)
            x, _ = example.output.shape
            max_x = max(max_x, x)
        # include test too
        x, _ = self.test_example.input.shape
        max_x = max(max_x, x)
        x, _ = self.test_example.output.shape
        max_x = max(max_x, x)
        return max_x

    def max_width(self) -> int:
        max_y = 0
        for example in self.train_examples:
            _, y = example.input.shape
            max_y = max(max_y, y)
            _, y = example.output.shape
            max_y = max(max_y, y)
        # include test too
        _, y = self.test_example.input.shape
        max_y = max(max_y, y)
        _, y = self.test_example.output.shape
        max_y = max(max_y, y)
        return max_y

    def __repr__(self) -> str:
        return f"Task(train={self.train_examples}, test={self.test_example})"

    def serialize(self) -> dict:
        return {
            "train": [train.serialize() for train in self.train_examples],
            "test": [self.test_example.serialize()],
            "name": self.name,
        }

    def __hash__(self) -> int:
        return hash((tuple(train for train in self.train_examples), self.test_example))

    @classmethod
    def deserialize(cls, data: dict, test: bool = False) -> "Task":
        assert len(data["test"]) == 1, "Only one test example is allowed"
        train = [Example.deserialize(train) for train in data["train"]]
        test = Example.deserialize(data["test"][0], test=test)
        return cls(train_examples=train, test_example=test, name=data.get("name", ""))

    @classmethod
    def read_tasks_from_dict(cls, data: dict, test: bool = False) -> List["Task"]:
        tasks = []
        for test_data in data["test"]:
            task = cls.deserialize(
                {"train": data["train"], "test": [test_data], "name": data.get("name", "")},
                test=test,
            )
            tasks.append(task)
        return tasks

    def entropy(self) -> float:
        """return the entropy of the outputs"""
        outputs = [example.output.flatten() for example in self.train_examples]
        outputs.append(self.test_example.output.flatten())
        vocabulary = np.unique(np.concatenate(outputs)).tolist()
        # find max output length
        max_output_length = max([len(output) for output in outputs])
        probs = np.zeros((len(vocabulary), max_output_length))
        # get the probes for each integer of each index
        for i, output in enumerate(outputs):
            for j, value in enumerate(output):
                index_of_value = vocabulary.index(value)
                probs[index_of_value, j] += 1

        # normalize
        probs = probs / probs.sum(axis=0)
        # get the entropy
        entropy = -np.sum(probs * np.log(probs + 1e-9), axis=0)

        # mean entropy
        return np.mean(entropy)


@dataclasses.dataclass
class TaskWithDescription(Task):
    description: str = ""


def read_tasks_from_folder(task_folder: str, test: bool = False) -> List[Task]:
    """
    Read tasks from a folder
    """
    all_tasks = []
    for file in glob.glob(f"{task_folder}/*.json"):
        basename = os.path.basename(file)
        idx = basename.replace(".json", "")
        tasks = read_tasks_from_file(file, test=test)
        for i, task in enumerate(tasks):
            task.name = idx + "-" + str(i)
        all_tasks += tasks
    return all_tasks


def read_tasks_from_single_file(
    challenge_file: str, test: bool = False, solution_file: Optional[str] = None
) -> List[Task]:
    """
    Read tasks from a single file
    """
    with open(challenge_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if solution_file is not None:
        test = False
        with open(solution_file, "r", encoding="utf-8") as handle:
            solutions = json.load(handle)
            for key, value in solutions.items():
                for idx, solution in enumerate(value):
                    data[key]["test"][idx]["output"] = solution

    all_tasks = []
    for task_name, subtasks in data.items():
        parsed_tasks = Task.read_tasks_from_dict(subtasks, test=test)
        for i, task in enumerate(parsed_tasks):
            task.name = task_name + "-" + str(i)
            all_tasks.append(task)

    return all_tasks


def read_tasks_from_file(task_file: str, test: bool = False) -> List[Task]:
    """
    Read tasks from a file
    """
    with open(task_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    return Task.read_tasks_from_dict(data, test=test)


def make_submission(
    tasks: List[Task],
    predictions: List[List[Grid]],
    path: Optional[str] = None,
    number_of_attempts: int = 2,
) -> dict:
    """
    Make a submission
    """
    assert len(tasks) == len(predictions), "Number of tasks and predictions should be the same"

    # sort by task_name alphabetically to ensure order of subtasks
    indices = np.argsort([task.name for task in tasks])
    tasks = [tasks[i] for i in indices]
    predictions = [predictions[i] for i in indices]
    # get the submissions
    submissions = {}
    for task, prediction in zip(tasks, predictions):
        task_name, task_no = task.name.split("-")
        task_no = int(task_no)
        if task_name not in submissions:
            submissions[task_name] = []

        assert len(prediction) == number_of_attempts, "Number of attempts should be the same"
        attempts = {f"attempt_{j+1}": to_list(pred) for j, pred in enumerate(prediction)}
        while len(submissions[task_name]) <= task_no:
            submissions[task_name].append({"attempt_1": [[0]], "attempt_2": [[0]]})

        submissions[task_name][task_no] = attempts

    if path is not None:
        with open(path, "w") as handle:
            json.dump(submissions, handle)

    return submissions


if __name__ == "__main__":
    arc_path = "/kaggle/input/arc-prize-2024/"
    tasks = read_tasks_from_single_file(arc_path + "arc-agi_training_challenges.json")
    print(tasks[0])
    tasks = read_tasks_from_single_file(arc_path + "arc-agi_evaluation_challenges.json", test=True)
    print(tasks[0])

    tasks = read_tasks_from_single_file(
        arc_path + "arc-agi_evaluation_challenges.json",
        test=True,
        solution_file=arc_path + "arc-agi_evaluation_solutions.json",
    )

    print(tasks[0])
