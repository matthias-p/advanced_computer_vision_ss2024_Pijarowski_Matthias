"""Implementation of a decision tree"""

from pathlib import Path

import numpy as np
from load_iris import loat_data

FEATURE_NAMES = [
    "Kelchblattlänge",
    "Kelchblattdicke",
    "Blütenblattlänge",
    "Blütenblattdicke",
]
CLASS_NAME_MAP = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}


class TreeNode:
    """Represents a node in the decision tree"""
    def predict(self, x: list[float]) -> tuple[int, float]:
        """Predict the class and probability from x by traversing the tree"""
        raise NotImplementedError

    def print(self, depth: int = 0) -> None:
        """Print the tree recursively"""
        raise NotImplementedError


class LeafNode(TreeNode):
    """Leaf Node"""

    def __init__(self, output_class: int, confidence: float) -> None:
        self.output_class = output_class
        self.confidence = confidence

    def predict(self, x: list[float]) -> tuple[int, float]:
        return self.output_class, self.confidence

    def print(self, depth: int = 0) -> None:
        print(
            " " * depth, "Leaf", CLASS_NAME_MAP.get(self.output_class), self.confidence
        )


class InnerNode(TreeNode):
    """Inner Node with left and right child"""

    def __init__(
        self,
        feature_index: int,
        split_value: float,
        left_child: TreeNode,
        right_child: TreeNode,
    ) -> None:
        self.feature_index = feature_index
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child

    def predict(self, x: list[float]) -> tuple[int, float]:
        if x[self.feature_index] <= self.split_value:
            return self.left_child.predict(x)
        return self.right_child.predict(x)

    def print(self, depth: int = 0) -> None:
        print(
            " " * depth,
            "Inner",
            FEATURE_NAMES[self.feature_index],
            "<",
            self.split_value,
        )
        self.left_child.print(depth + 1)
        self.right_child.print(depth + 1)


DataPoints = list[tuple[list[float], int]]


def count_occurances_of_class(data_points: DataPoints, class_: int) -> int:
    """Return how often class occurs in data_points"""
    occurances = 0
    for _, y in data_points:
        if y == class_:
            occurances += 1
    return occurances


def get_unique_classes(data_points: DataPoints) -> list[int]:
    """Return a list of all the classes that occur in data_points"""
    unique_classes = set()
    for _, y in data_points:
        unique_classes.add(y)
    return list(unique_classes)


def gini_index(data_points: DataPoints, classes: list[int]) -> float:
    """Calculate gini index for given classes"""
    gini = 0
    for class_ in classes:
        p_class = count_occurances_of_class(data_points, class_) / len(data_points)
        gini += p_class**2
    return 1 - gini


def split_data(
    data_points: DataPoints, feature_index: int, split_value: float
) -> tuple[DataPoints, DataPoints]:
    """Split data according to feature and split value into left and right"""
    left_data_points: DataPoints = []
    right_data_points: DataPoints = []
    for x, y in data_points:
        if x[feature_index] <= split_value:
            left_data_points.append((x, y))
        else:
            right_data_points.append((x, y))
    return left_data_points, right_data_points


def suggest_split_values(data_points: DataPoints, feature_index: int) -> float:
    """Return the best values to split the data on"""
    feature_values = [x[feature_index] for x, _ in data_points]
    return np.median(feature_values)  # type: ignore


def select_best_split(
    data_points: DataPoints, feature_indices: list[int]
) -> tuple[int, float]:
    """Select the best featue to split the data on and return its value"""
    best_feature_index = 0
    best_split_value = 0.0
    best_impurity = float("inf")

    for feature_index in feature_indices:
        suggested_split_value = suggest_split_values(data_points, feature_index)
        left, right = split_data(data_points, feature_index, suggested_split_value)
        impurity = gini_index(left, get_unique_classes(left)) + gini_index(
            right, get_unique_classes(right)
        )

        if impurity < best_impurity:
            best_impurity = impurity
            best_feature_index = feature_index
            best_split_value = suggested_split_value

    return best_feature_index, best_split_value


def get_most_probable_class(data_points: DataPoints) -> tuple[int, float]:
    """Return the most frequent class in data_points and its probability"""
    unique_classes = get_unique_classes(data_points)
    class_occurance_tuples = []
    for class_ in unique_classes:
        occurances = count_occurances_of_class(data_points, class_)
        class_occurance_tuples.append((class_, occurances))
    class_, occurances = sorted(class_occurance_tuples, key=lambda x: x[1])[-1]
    return class_, occurances / len(data_points)


def build_tree(data_points: DataPoints) -> TreeNode:
    """Build the decision tree recursively"""
    unique_classes = get_unique_classes(data_points)

    if (len(unique_classes) == 1) or (len(data_points) <= 5):
        class_, proba = get_most_probable_class(data_points)
        return LeafNode(class_, proba)

    best_feature_index, best_split_value = select_best_split(data_points, [0, 1, 2, 3])
    left_data, right_data = split_data(
        data_points, best_feature_index, best_split_value
    )

    left_child = build_tree(left_data)
    right_child = build_tree(right_data)
    return InnerNode(best_feature_index, best_split_value, left_child, right_child)


def main():  # pylint: disable=missing-function-docstring
    file_dir = Path(__file__).parent
    x, y = loat_data(file_dir / "iris_data.csv")
    data_points: DataPoints = []
    for x_, y_ in zip(x, y):
        data_points.append((x_, y_))
    decision_tree = build_tree(data_points)
    decision_tree.print()

    hits = 0
    for x, y in data_points:
        c, _ = decision_tree.predict(x)
        if c == y:
            hits += 1
    print(hits / len(data_points))


if __name__ == "__main__":
    main()
