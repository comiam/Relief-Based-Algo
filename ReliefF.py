from random import randrange

import numpy as np


class ReliefF:
    def __init__(self, data: np.ndarray, classes: np.ndarray, iterations: int, knn: int = 10):
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        if knn < 0:
            raise ValueError("invalid k count of neighbours!")

        self.data = data
        self.classes = classes
        self.iter = iterations
        self.k = knn

    def fit(self) -> np.ndarray:
        w = np.array([0.] * self.data.shape[1])

        probs = self._class_frequencies(self.classes)
        # print(np.fromiter(probs.values(), dtype=float))

        for i in range(self.iter):
            random = randrange(self.data.shape[0])

            knn = self._nn(random)
            hit = knn[self.classes[random]]
            del knn[self.classes[random]]
            miss = knn

            for k in range(self.data.shape[1]):
                w[k] += (
                        np.array([
                                 (probs[c] / (1 - probs[self.classes[random]])) *
                                 np.array([
                                        self._diff_value(k, random, m_idx)
                                        for m_idx in miss[c]
                                 ]).sum()
                                 for c in miss.keys()
                        ]).sum() / (self.iter * self.k)
                        - np.array([self._diff_value(k, random, h_idx) for h_idx in hit]).sum() / (self.iter * self.k)
                )
        return w

    def _nn(self, ind: int) -> dict[int, np.ndarray]:
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(ind, r[0]),
                    enumerate(self.data)
                )
            )
        )

        sorted_inds = dist.argsort()
        oy = self.classes[sorted_inds]

        return \
            {
                # ignore first c class because of pivot object in same vector and its distance equals 0
                c: sorted_inds[oy == c][slice(0, self.k) if c != self.classes[ind] else slice(1, self.k + 1)]
                for c in self.classes
            }

    def _diff(self, a: int, b: int) -> float:
        return np.array([self._diff_value(c, a, b) for c in range(self.data.shape[1])]).sum()

    def _diff_value(self, feature_index: int, a: int, b: int) -> float:
        a_val = self.data[a, feature_index]
        b_val = self.data[b, feature_index]
        na = np.isnan(a_val)
        nb = np.isnan(b_val)

        if not na and not nb:
            return self._diff_none_nan(feature_index, a, b)
        elif na and not nb:
            return self._diff_one_nan(feature_index, b, a)
        elif nb and not na:
            return self._diff_one_nan(feature_index, a, b)
        elif nb and na:
            return self._diff_both_nan(feature_index, a, b)

    def _diff_none_nan(self, feature_index: int, a: int, b: int) -> float:
        rmax = np.amax(self.data[:, feature_index])
        rmin = np.amin(self.data[:, feature_index])

        return np.abs(self.data[a, feature_index] - self.data[b, feature_index]) / (rmax if (rmax - rmin) == 0 else (rmax - rmin))

    def _diff_both_nan(self, feature_index: int, a: int, b: int) -> float:
        class0 = self.classes[a]
        class1 = self.classes[b]

        return 1. - np.array([
            self._frequency(v, self.data[self.classes == class0, feature_index]) *
            self._frequency(v, self.data[self.classes == class1, feature_index])
            for v in set(self.data[:, feature_index])
            if not np.isnan(v)
        ]).sum()

    def _diff_one_nan(self, feature_index: int, known_val_ind: int, unknown_val_ind: int) -> float:
        known_val = self.data[known_val_ind, feature_index]
        unknown_val_class = self.classes[unknown_val_ind]

        return 1. - self._frequency(known_val, self.data[self.classes == unknown_val_class, feature_index])

    def _class_frequencies(self, classes: np.ndarray) -> dict[int, float]:
        return {c: self._frequency(c, classes) for c in set(classes)}

    def _frequency(self, value: float, vector: np.ndarray) -> float:
        if vector.shape[0] == 0:
            return 0

        return (vector == value).sum() / len(vector)
