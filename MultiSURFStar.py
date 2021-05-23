from random import randrange

import numpy as np

from SURFStar import SURFStar


class MultiSURFStar(SURFStar):
    def __init__(self, iterations: int):
        super().__init__(iterations)

    def fit(self, data: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        w = np.array([0.] * data.shape[1])
        dists = self.dist_matrix(data)
        self.threshold = dists.flatten()[dists.flatten() != 0.0].mean()
        std_val = dists.flatten()[dists.flatten() != 0.0].std()
        D = std_val / 2

        for i in range(self.iter):
            random = randrange(data.shape[0])

            nn = self._nn(data, classes, random, self.threshold, D)
            nn_nearest = nn[0]
            nn_far = nn[1]

            hit = nn_nearest[classes[random]]
            del nn_nearest[classes[random]]
            miss = nn_nearest

            hit_far = nn_far[classes[random]]
            del nn_far[classes[random]]
            miss_far = nn_far

            for k in range(data.shape[1]):
                # for nearest
                w[k] += (
                        np.array([
                            self._diff_value(data, classes, k, random, m_idx)
                            for c in miss.keys() for m_idx in miss[c]
                        ]).sum() / (self.iter * self.k)
                        - np.array([self._diff_value(data, classes, k, random, h_idx) for h_idx in hit]).sum() / (
                                self.iter * self.k)
                )
                # for far instances
                w[k] += (
                        np.array([
                            self._diff_value(data, classes, k, random, m_idx)
                            for c in miss_far.keys() for m_idx in miss_far[c]
                        ]).sum() / (self.iter * self.k)
                        + np.array([self._diff_value(data, classes, k, random, h_idx) for h_idx in hit_far]).sum() / (
                                self.iter * self.k)
                )
        return w

    def _nn(self, data: np.ndarray, classes: np.ndarray, ind: int, threshold: float, std: float) -> list[dict[int, np.ndarray]]:
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(data, classes, ind, r[0]),
                    enumerate(data)
                )
            )
        )

        dist_near = dist[dist <= self.threshold - std]
        dist_far = dist[dist > self.threshold + std]

        sorted_inds = dist_near.argsort()
        oy = classes[sorted_inds]

        sorted_inds_far = dist_far.argsort()
        oy_far = classes[sorted_inds_far]

        return \
            [{
                # ignore first c class because of pivot object in same vector and its distance equals 0
                c: sorted_inds[oy == c] if c != classes[ind] else sorted_inds[oy == c][1:len(sorted_inds[oy == c]) + 1]
                for c in classes
            }, {
                c: sorted_inds_far[oy_far == c] for c in classes
            }]

    def dist_matrix(self, data: np.ndarray) -> np.ndarray:
        d = np.zeros((data.shape[0], data.shape[0]), dtype=float)
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                d[i][j] = np.abs(data[i] - data[j]).sum()

        return d
