"""Implementation of a decision tree"""

from pathlib import Path
import random

import numpy as np
from load_iris import loat_data


class TreeNode:
    """Represents a node in the decision tree"""
    def __init__(self, feature_names: dict, class_names: dict) -> None:
        self.feature_names = feature_names
        self.class_names = class_names

    def predict(self, x: list[float]) -> tuple[int, float]:
        """Predict the class and probability from x by traversing the tree"""
        raise NotImplementedError

    def print(self, depth: int = 0) -> None:
        """Print the tree recursively"""
        raise NotImplementedError


class LeafNode(TreeNode):
    """Leaf Node"""

    def __init__(
        self,
        output_class: int,
        confidence: float,
        feature_names: dict,
        class_names: dict,
    ) -> None:
        super().__init__(feature_names, class_names)
        self.output_class = output_class
        self.confidence = confidence

    def predict(self, x: list[float]) -> tuple[int, float]:
        return self.output_class, self.confidence

    def print(self, depth: int = 0) -> None:
        print(
            " " * depth,
            "Leaf",
            self.class_names.get(self.output_class, self.output_class),
            self.confidence,
        )


class InnerNode(TreeNode):
    """Inner Node with left and right child"""

    def __init__(
        self,
        feature_index: int,
        split_value: float,
        left_child: TreeNode,
        right_child: TreeNode,
        feature_names: dict,
        class_names: dict,
    ) -> None:
        super().__init__(feature_names, class_names)
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
            self.feature_names.get(self.feature_index),
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


def suggest_split_values(data_points: DataPoints, feature_index: int) -> list[float]:
    """Return the best values to split the data on"""
    unique_classes = get_unique_classes(data_points)
    class_0_features = []
    for x, y in data_points:
        if y == unique_classes[0]:
            class_0_features.append(x[feature_index])

    feature_values = [x[feature_index] for x, _ in data_points]
    return [np.median(feature_values), np.mean(feature_values), np.max(class_0_features)]  # type: ignore


def select_best_split(
    data_points: DataPoints, feature_indices: list[int], evaluate_on_sample_size: None | float = None
) -> tuple[int, float]:
    """Select the best featue to split the data on and return its value"""
    best_feature_index = 0
    best_split_value = 0.0
    best_impurity = float("inf")

    if evaluate_on_sample_size is None:
        samples_to_evaluate_on = data_points
    else:
        samples_to_evaluate_on = random.choices(data_points, k=int(len(data_points) * evaluate_on_sample_size))

    for feature_index in feature_indices:
        suggested_split_values = suggest_split_values(samples_to_evaluate_on, feature_index)
        for suggestd_split_value in suggested_split_values:
            left, right = split_data(samples_to_evaluate_on, feature_index, suggestd_split_value)
            impurity = gini_index(left, get_unique_classes(left)) + gini_index(
                right, get_unique_classes(right)
            )

            if impurity < best_impurity:
                best_impurity = impurity
                best_feature_index = feature_index
                best_split_value = suggestd_split_value

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


def build_tree(data_points: DataPoints, feature_names: dict, class_names: dict) -> TreeNode:
    """Build the decision tree recursively"""
    unique_classes = get_unique_classes(data_points)

    if (len(unique_classes) == 1) or (len(data_points) <= 5):
        class_, proba = get_most_probable_class(data_points)
        return LeafNode(class_, proba, feature_names, class_names)

    best_feature_index, best_split_value = select_best_split(
        data_points, list(range(len(data_points[0][0])))
    )
    left_data, right_data = split_data(
        data_points, best_feature_index, best_split_value
    )

    left_child = build_tree(left_data, feature_names, class_names)
    right_child = build_tree(right_data, feature_names, class_names)
    return InnerNode(
        best_feature_index,
        best_split_value,
        left_child,
        right_child,
        feature_names,
        class_names,
    )

def main():  # pylint: disable=missing-function-docstring
    file_dir = Path(__file__).parent
    x, y = loat_data(file_dir / "iris_data.csv")
    data_points: DataPoints = []
    for x_, y_ in zip(x, y):
        data_points.append((x_, y_))

    feature_names = {
        0: "Kelchblattlänge",
        1: "Kelchblattdicke",
        2: "Blütenblattlänge",
        3: "Blütenblattdicke",
    }
    class_names = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

    decision_tree = build_tree(data_points, feature_names, class_names)
    decision_tree.print()

    hits = 0
    for x, y in data_points:
        c, _ = decision_tree.predict(x)
        if c == y:
            hits += 1
    print(hits / len(data_points))


if __name__ == "__main__":
    main()
