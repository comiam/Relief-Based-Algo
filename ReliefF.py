from random import randrange

import numpy as np


# First upgrade of Relief algorithm, now there are non binary classification and used k NM and NH
# https://link.springer.com/content/pdf/10.1023/A:1025667309714.pdf
class ReliefF:
    def __init__(self, iterations: int = 100, knn: int = 10):
        if knn <= 0:
            raise ValueError("invalid k count of neighbours!")

        self.iter = iterations
        self.k = knn

    def fit(self, data: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        w = np.array([0.] * data.shape[1])

        probs = self._class_frequencies(classes)
        # rmax = np.array([np.amax(data[:, f]) for f in range(data.shape[1])])
        # rmin = np.array([np.amin(data[:, f]) for f in range(data.shape[1])])
        # print(np.fromiter(probs.values(), dtype=float))

        for i in range(self.iter):
            random = randrange(data.shape[0])

            knn = self._nn(data, classes, random)
            hit = knn[classes[random]]
            del knn[classes[random]]
            miss = knn

            for k in range(data.shape[1]):
                w[k] += (
                        np.array([
                            (probs[c] / (1 - probs[classes[random]])) *
                            np.array([
                                self._diff_value(data, classes, k, random, m_idx)
                                for m_idx in miss[c]
                            ]).sum()
                            for c in miss.keys()
                        ]).sum() / (self.iter * self.k)
                        - np.array([self._diff_value(data, classes, k, random, h_idx) for h_idx in hit]).sum() / (
                                    self.iter * self.k)
                )
        return w

    def _nn(self, data: np.ndarray, classes: np.ndarray, ind: int) -> dict[int, np.ndarray]:
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(data, classes, ind, r[0]),
                    enumerate(data)
                )
            )
        )

        sorted_inds = dist.argsort()
        oy = classes[sorted_inds]

        return \
            {
                # ignore first c class because of pivot object in same vector and its distance equals 0
                c: sorted_inds[oy == c][slice(0, self.k) if c != classes[ind] else slice(1, self.k + 1)]
                for c in classes
            }

    def _diff(self, data: np.ndarray, classes: np.ndarray, a: int, b: int) -> float:
        return np.array([self._diff_value(data, classes, c, a, b) for c in range(data.shape[1])]).sum()

    def _diff_value(self, data: np.ndarray, classes: np.ndarray, feature_index: int, a: int, b: int) -> float:
        a_val = data[a, feature_index]
        b_val = data[b, feature_index]
        na = np.isnan(a_val)
        nb = np.isnan(b_val)

        if not na and not nb:
            return self._diff_none_nan(data, feature_index, a, b)
        elif na and not nb:
            return self._diff_one_nan(data, classes, feature_index, b, a)
        elif nb and not na:
            return self._diff_one_nan(data, classes, feature_index, a, b)
        elif nb and na:
            return self._diff_both_nan(data, classes, feature_index, a, b)

    def _diff_none_nan(self, data: np.ndarray, feature_index: int, a: int, b: int) -> float:
        rmax = np.amax(data[:, feature_index])
        rmin = np.amin(data[:, feature_index])

        return np.abs(data[a, feature_index] - data[b, feature_index]) / (1 if (rmax - rmin) == 0 else (rmax - rmin))

    def _diff_both_nan(self, data: np.ndarray, classes: np.ndarray, feature_index: int, a: int, b: int) -> float:
        class0 = classes[a]
        class1 = classes[b]

        return 1. - np.array([
            self._frequency(v, data[classes == class0, feature_index]) *
            self._frequency(v, data[classes == class1, feature_index])
            for v in set(data[:, feature_index])
            if not np.isnan(v)
        ]).sum()

    def _diff_one_nan(self, data: np.ndarray, classes: np.ndarray, feature_index: int, known_val_ind: int,
                      unknown_val_ind: int) -> float:
        known_val = data[known_val_ind, feature_index]
        unknown_val_class = classes[unknown_val_ind]

        return 1. - self._frequency(known_val, data[classes == unknown_val_class, feature_index])

    def _class_frequencies(self, classes: np.ndarray) -> dict[int, float]:
        return {c: self._frequency(c, classes) for c in set(classes)}

    def _frequency(self, value: float, vector: np.ndarray) -> float:
        if vector.shape[0] == 0:
            return 0

        return (vector == value).sum() / len(vector)
