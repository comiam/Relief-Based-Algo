import numpy as np

from SURF import SURF


# https://biodatamining.biomedcentral.com/articles/10.1186/1756-0381-5-20
class MultiSURF(SURF):
    def __init__(self, iterations: int):
        super().__init__(iterations, 0.0, 1)

    def fit(self, data: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        dists = self.dist_matrix(data)
        self.threshold = dists.flatten()[dists.flatten() != 0.0].mean()
        std_val = dists.flatten()[dists.flatten() != 0.0].std()
        D = std_val / 2

        w = np.array([0.] * data.shape[1])

        for i in range(self.iter):
            for si in range(data.shape[0]):
                w += self.compute_margin(data, classes, self.threshold - D, si)

        return w

    def compute_margin(self, data: np.ndarray, classes: np.ndarray, thr: float, i: int) -> np.ndarray:
        knn = self._nn(data, classes, i, thr)
        hit = knn[classes[i]]
        del knn[classes[i]]
        miss = knn

        return np.array([
            (
                    np.array([
                        np.array([
                            self._diff_value(data, classes, k, i, m_idx)
                            for m_idx in miss[c]
                        ]).sum()
                        for c in miss.keys()
                    ]).sum() / (data.shape[0] * self.iter)
                    - np.array([self._diff_value(data, classes, k, i, h_idx) for h_idx in hit]).sum()
                    / (data.shape[0] * self.iter)
            )
            for k in range(data.shape[1])
        ])

    def _nn(self, data: np.ndarray, classes: np.ndarray, ind: int, threshold: float) -> dict[int, np.ndarray]:
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(data, classes, ind, r[0]),
                    enumerate(data)
                )
            )
        )

        dist = dist[dist <= threshold]

        sorted_inds = dist.argsort()
        oy = classes[sorted_inds]

        return \
            {
                # ignore first c class because of pivot object in same vector and its distance equals 0
                c: sorted_inds[oy == c] if c != classes[ind] else sorted_inds[oy == c][1:len(sorted_inds[oy == c]) + 1]
                for c in classes
            }

    def dist_matrix(self, data: np.ndarray) -> np.ndarray:
        d = np.zeros((data.shape[0], data.shape[0]), dtype=float)
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                d[i][j] = np.abs(data[i] - data[j]).sum()

        return d
