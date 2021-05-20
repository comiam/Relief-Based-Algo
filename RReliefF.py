from random import randrange

import numpy as np

from ReliefF import ReliefF


# Adapted for regression ReliefF http://www.clopinet.com/isabelle/Projects/reading/robnik97-icml.pdf
class RReliefF(ReliefF):
    def __init__(self, iterations: int, knn: int = 10, sigma: float = 1.0):
        super().__init__(iterations, knn)

        if sigma < 0:
            raise ValueError("invalid kernel width!")

        self.sigma = sigma

    def fit(self, data: np.ndarray, classes: np.ndarray):
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")
        
        ndcda = np.array([0.] * data.shape[1])
        nda = np.array([0.] * data.shape[1])
        ndc = 0.0
        ds = self.calc_factors()

        for i in range(self.iter):
            random = randrange(data.shape[0])
            knn = self._nn(data, classes, random)

            for q in range(self.k):
                d = ds[q]
                diff_label = np.abs(classes[random] - classes[knn[q]])
                ndc += diff_label * d

                for k in range(data.shape[1]):
                    nda_incr = self._diff_value(data, classes, k, random, knn[q]) * d
                    nda[k] += nda_incr
                    ndcda[k] += diff_label * nda_incr

        return ndcda / ndc - (nda - ndcda) / (self.iter - ndc)

    def _nn(self, data: np.ndarray, classes: np.ndarray, j: int) -> np.ndarray:
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(data, classes, j, r),
                    range(data.shape[0])
                )
            )
        )

        return dist.argsort()[1:self.k + 1]

    def calc_factors(self):
        d1 = np.exp(-((np.arange(self.k) + 1.) / self.sigma) ** 2)

        return d1 / d1.sum()
