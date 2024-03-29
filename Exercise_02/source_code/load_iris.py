"""Dataloader for iris dataset in sklearn format"""

import csv
from pathlib import Path


class_name_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}


def loat_data(csv_path: str | Path) -> tuple[list[list[float]], list[int]]:
    """Load iris csv from string and parse it according to spec"""
    with open(csv_path, "r", encoding="utf-8") as fp:
        reader = csv.reader(fp, delimiter=";")

        x = []
        y = []
        for row in reader:
            x_features = [float(value) for value in row[:-1]]
            x.append(x_features)
            y.append(class_name_map.get(row[-1]))

    return x, y


def main():  # pylint: disable=missing-function-docstring
    x, y = loat_data(
        "/home/m/Documents/advanced_computer_vision_ss2024_Pijarowski_Matthias/Exercise_02/source_code/iris_data.csv"
    )
    print(len(x))
    print(len(y))
    print(y)
    print(x)


if __name__ == "__main__":
    main()
