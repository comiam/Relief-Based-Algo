from random import randrange

import numpy as np

from ReliefF import ReliefF


#https://biodatamining.biomedcentral.com/articles/10.1186/1756-0381-2-5
class SURF(ReliefF):
    def __init__(self, iterations: int, threshold: float = 0.2, knn: int = 1):
        super().__init__(iterations, knn)

        if knn < 0:
            raise ValueError("invalid distance threshold!")

        self.threshold = threshold

    def fit(self, data: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        w = np.array([0.] * data.shape[1])
        probs = self._class_frequencies(classes)

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

        dist = dist[dist <= self.threshold]

        sorted_inds = dist.argsort()
        oy = classes[sorted_inds]

        return \
            {
                # ignore first c class because of pivot object in same vector and its distance equals 0
                c: sorted_inds[oy == c] if c != classes[ind] else sorted_inds[oy == c][1:len(sorted_inds[oy == c]) + 1]
                for c in classes
            }

    def set_threshold(self, t: float):
        self.threshold = t

    # can use as threshold value
    def compute_avg_distance(self, data: np.ndarray, classes: np.ndarray) -> float:
        sum = 0.0
        count = 0
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                sum += self._diff(data, classes, i, j)
                count += 1

        return sum / count
