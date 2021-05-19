from math import log

import numpy as np


def find_interval(value: float, intervals: list[tuple[float, float]]):
    it = 0
    for i in intervals:
        if i[0] <= value < i[1]:
            return it
        elif value == i[1] and i == intervals[-1]:
            return it
        it += 1
    return np.NaN


def split_classes(classes: np.ndarray, delete_last: int = 0) -> np.ndarray:
    bins = 1 + log(classes.shape[0], 2)
    min = np.amin(classes)
    max = np.amax(classes)
    length = (max - min) / int(bins)

    ints = [(min + length * i, min + length * (i + 1)) for i in range(int(bins))]

    if delete_last != 0:
        last_end_range = ints[-1][1]
        ints = ints[:len(ints)-delete_last]
        last_start_range = ints[-1][0]
        del ints[-1]
        ints.append((last_start_range, last_end_range))

    splitted = np.array([find_interval(c, ints) for c in classes])

    return splitted
