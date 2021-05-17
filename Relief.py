from random import randrange

import numpy as np


# Base version of Relief with binary classification https://arxiv.org/pdf/1711.08421v1.pdf
class Relief:
    def __init__(self, iterations: int = 100):
        self.iter = iterations
        self.k = 1

    def fit(self, data: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        w = np.array([0.] * data.shape[1])

        for i in range(self.iter):
            random = randrange(data.shape[0])

            h, m = self._nn(data, classes, random)

            w += np.array([
                self._diff(data, f, data[random], m) / self.iter
                - self._diff(data, f, data[random], h) / self.iter for f in range(len(w))
            ])

        return w

    def _diff(self, data: np.ndarray, feature_index: int, a: np.ndarray, b: np.ndarray) -> float:
        rmax = np.amax(data[:, feature_index])
        rmin = np.amin(data[:, feature_index])

        return np.abs(a[feature_index] - b[feature_index]) / (rmax - rmin)

    def _nn(self, data: np.ndarray, classes: np.ndarray, ind: int) -> tuple[np.ndarray, np.ndarray]:
        selected = data[ind]

        dist = np.sum(np.array(
            [np.abs(selected[c] - data[:, c]) for c in range(len(selected))]
        ).T, axis=1)

        odata = data[dist.argsort()]
        oy = classes[dist.argsort()]

        h = odata[oy == classes[ind]][0:1]
        m = odata[oy != classes[ind]][0]

        h = h[1] if h.shape[0] > 1 else h[0]

        return h, m
