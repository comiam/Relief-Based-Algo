from random import randrange

import numpy as np

from ReliefF import ReliefF


class RReliefF(ReliefF):
    def __init__(self, data: np.ndarray, classes: np.ndarray, iterations: int, knn: int = 10, sigma: float = 1.0):
        super().__init__(data, classes, iterations, knn)
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        if knn < 0:
            raise ValueError("invalid k count of neighbours!")

        self.sigma = sigma

    def fit(self):
        ndcda = np.array([0.] * self.data.shape[1])
        nda = np.array([0.] * self.data.shape[1])
        ndc = 0.0
        ds = self.calc_factors()

        for i in range(self.iter):
            random = randrange(self.data.shape[0])
            knn = self._nn(random)

            for q in range(self.k):
                d = ds[q]
                diff_label = np.abs(self.classes[random] - self.classes[knn[q]])
                ndc += diff_label * d

                for k in range(self.data.shape[1]):
                    nda_incr = self._diff_value(k, random, knn[q]) * d
                    nda[k] += nda_incr
                    ndcda[k] += diff_label * nda_incr

        return ndcda / ndc - (nda - ndcda) / (self.iter - ndc)

    def _nn(self, j: int) -> np.ndarray:
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(j, r),
                    range(self.data.shape[0])
                )
            )
        )

        return dist.argsort()[1:self.k + 1]

    def calc_factors(self):
        d1 = np.exp(-((np.arange(self.k) + 1.) / self.sigma) ** 2)

        return d1 / d1.sum()
