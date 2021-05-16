from random import randrange

import numpy as np


class Relief:
    def __init__(self, data: np.ndarray, classes: np.ndarray, iterations: int):
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        self.data = data
        self.classes = classes
        self.iter = iterations
        self.k = 1

    def fit(self) -> np.ndarray:
        w = np.array([0.] * self.data.shape[1])

        for i in range(self.iter):
            random = randrange(self.data.shape[0])

            h, m = self._nn(self.data, self.classes, random)

            w += np.array([
                self._diff(f, self.data[random], m) / self.iter
                - self._diff(f, self.data[random], h) / self.iter for f in range(len(w))
            ])

        return w

    def _diff(self, feature_index: int, a: np.ndarray, b: np.ndarray) -> float:
        rmax = np.amax(self.data[:, feature_index])
        rmin = np.amin(self.data[:, feature_index])

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
