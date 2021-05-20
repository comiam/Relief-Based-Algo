from random import randrange

import numpy as np

from SURF import SURF


# https://link.springer.com/content/pdf/10.1007%2F978-3-642-12211-8_16.pdf
class SURFStar(SURF):
    def __init__(self, iterations: int, threshold: float = 0.2):
        super().__init__(iterations, threshold, 1)

    def fit(self, data: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        w = np.array([0.] * data.shape[1])
        probs = self._class_frequencies(classes)

        for i in range(self.iter):
            random = randrange(data.shape[0])

            nn = self._nn(data, classes, random)
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
                # for far instances
                w[k] += (
                        np.array([
                            -(probs[c] / (1 - probs[classes[random]])) *
                            np.array([
                                self._diff_value(data, classes, k, random, m_idx)
                                for m_idx in miss_far[c]
                            ]).sum()
                            for c in miss_far.keys()
                        ]).sum() / (self.iter * self.k)
                        + np.array([self._diff_value(data, classes, k, random, h_idx) for h_idx in hit_far]).sum() / (
                                self.iter * self.k)
                )
        return w

    def _nn(self, data: np.ndarray, classes: np.ndarray, ind: int) -> list[dict[int, np.ndarray]]:
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(data, classes, ind, r[0]),
                    enumerate(data)
                )
            )
        )

        dist_near = dist[dist <= self.threshold]
        dist_far = dist[dist > self.threshold]

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
