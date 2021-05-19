import functools
import os

import numba
import numpy as np
from multiprocessing import Pool

from numba import prange


def _nn(data: np.ndarray, classes: np.ndarray, rmax: np.ndarray, rmin: np.ndarray, ind: int, k: int) -> dict[int, np.ndarray]:
    with Pool(np.max([os.cpu_count()-1, 1])) as P:
        diff_list = P.map(functools.partial(_diff, data, classes, rmax, rmin, ind), enumerate(data))

    dist = np.array(list(diff_list))

    sorted_inds = dist.argsort()
    oy = classes[sorted_inds]

    return \
        {
            # ignore first c class because of pivot object in same vector and its distance equals 0
            c: sorted_inds[oy == c][slice(0, k) if c != classes[ind] else slice(1, k + 1)]
            for c in classes
        }


@numba.njit
def sort_dists(dists: np.ndarray) -> np.ndarray:
    return dists.argsort()


@numba.njit(parallel=True)
def _diff(data: np.ndarray, classes: np.ndarray, rmax: np.ndarray, rmin: np.ndarray, a: int, b: tuple) -> float:
    res = 0

    for c in prange(data.shape[1]):
        res += _diff_value(data, classes, c, rmax[c], rmin[c], a, b[0])

    return res


@numba.njit
def _diff_value(data: np.ndarray, classes: np.ndarray, feature_index: int, rmax: float, rmin: float, a: int, b: int) -> float:
    a_val = data[a, feature_index]
    b_val = data[b, feature_index]
    na = np.isnan(a_val)
    nb = np.isnan(b_val)

    if not na and not nb:
        return _diff_none_nan(data, feature_index, rmax, rmin, a, b)
    elif na and not nb:
        return _diff_one_nan(data, classes, feature_index, b, a)
    elif nb and not na:
        return _diff_one_nan(data, classes, feature_index, a, b)
    elif nb and na:
        return _diff_both_nan(data, classes, feature_index, a, b)


@numba.njit
def _diff_none_nan(data: np.ndarray, feature_index: int, rmax: float, rmin: float, a: int, b: int) -> float:
    return np.abs(data[a, feature_index] - data[b, feature_index]) / (1 if (rmax - rmin) == 0 else (rmax - rmin))


@numba.njit
def _diff_both_nan(data: np.ndarray, classes: np.ndarray, feature_index: int, a: int, b: int) -> float:
    class0 = classes[a]
    class1 = classes[b]

    return 1. - np.array([
        _frequency(v, data[classes == class0, feature_index]) *
        _frequency(v, data[classes == class1, feature_index])
        for v in set(data[:, feature_index])
        if not np.isnan(v)
    ]).sum()


@numba.njit
def _diff_one_nan(data: np.ndarray, classes: np.ndarray, feature_index: int, known_val_ind: int,
                  unknown_val_ind: int) -> float:
    known_val = data[known_val_ind, feature_index]
    unknown_val_class = classes[unknown_val_ind]

    return 1. - _frequency(known_val, data[classes == unknown_val_class, feature_index])


@numba.njit
def _class_frequencies(classes: np.ndarray) -> dict[int, float]:
    return {c: _frequency(c, classes) for c in set(classes)}


@numba.njit
def _frequency(value: float, vector: np.ndarray) -> float:
    if vector.shape[0] == 0:
        return 0

    return (vector == value).sum() / len(vector)
