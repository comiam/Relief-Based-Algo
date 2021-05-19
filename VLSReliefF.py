import random

import numpy as np
from numpy import inf

from ReliefF import ReliefF


class VLSReliefF(ReliefF):
    def __init__(self, iterations: int = 100, knn: int = 10):
        super().__init__(iterations, knn)

    def fit(self, data: np.ndarray, classes: np.ndarray, subseq_count: int, subseq_length: int) -> np.ndarray:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        w = np.array([-inf] * data.shape[1])

        subs = self._rand_parts(list(range(data.shape[1])), subseq_count, subseq_length)

        for seq in subs:
            data_copy = data.copy()
            np.delete(data_copy, np.array([i for i in range(data.shape[1]) if i not in seq]), axis=1)
            w_local = super().fit(data_copy, classes)

            for i in range(len(seq)):
                w[seq[i]] = np.maximum(w_local[i], w[seq[i]])

            del data_copy

        return w

    def _rand_parts(self, seq: list, n: int, l: int) -> list[list[int]]:
        result = []

        for i in range(n):
            ind = random.sample(seq, l)
            ind.sort()
            result.append(ind)

        set_l = [set(l) for l in result]
        final_set = set()
        for s in set_l:
            final_set = final_set.union(s)

        result.append(list(set(seq) - final_set))

        return result
