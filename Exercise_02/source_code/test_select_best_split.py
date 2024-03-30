import random
import time
from decision_tree import select_best_split

random.seed(42)


def generate_data_points(num_classes, num_features, num_samples):
    classes = list(range(num_classes))

    data_points = []
    for _ in range(num_samples):
        x_features = [random.random() for _ in range(num_features)]
        y = random.choice(classes)
        data_points.append((x_features, y))
    return data_points


def main():
    for num_samples in [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]:
        data_points = generate_data_points(3, 4, num_samples)
        t1 = time.time()
        select_best_split(data_points, list(range(4)))
        print(f"Num samples: {num_samples} / {num_samples}: {time.time() - t1:5f}")

        t1 = time.time()
        select_best_split(data_points, list(range(4)), evaluate_on_sample_size=0.25)
        print(f"Num samples: {num_samples * 0.25} / {num_samples}: {time.time() - t1:5f}")

        print()


if __name__ == "__main__":
    main()
