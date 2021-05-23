import numpy as np

from ReliefF import ReliefF

#https://biodatamining.biomedcentral.com/articles/10.1186/1756-0381-5-20
class SWRF(ReliefF):
    def __init__(self, iterations: int, knn: int = 1):
        super().__init__(iterations, knn)

    def fit(self, data: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        dists = self.dist_matrix(data)
        if hasattr(self, 't'):
            t_selected = self.t
        else:
            t_selected = dists.flatten()[dists.flatten() != 0.0].mean()

        if hasattr(self, 'u'):
            u_selected = self.u
        else:
            u_selected = dists.flatten()[dists.flatten() != 0.0].std()

        dists = self.fill_matrix(dists)

        sigmoid = self.sigmoid_matrix(dists, t_selected, u_selected)

        w = np.array([0.] * data.shape[1])

        for i in range(self.iter):
            for k in range(data.shape[1]):
                w[k] += np.mean(np.array([self.compute_margin(data, classes, sigmoid, si, k)
                                          for si in range(data.shape[0])]))

        return w

    def compute_margin(self, data: np.ndarray, classes: np.ndarray, sigm: np.ndarray, i: int, k: int) -> float:
        knn = self._nn(data, classes, i)
        hit = knn[classes[i]]
        del knn[classes[i]]
        miss = knn

        return (
                       np.array([
                           self._diff_value(data, classes, k, i, m_idx) * sigm[i][m_idx]
                           for c in miss.keys() for m_idx in miss[c]
                       ]).sum()
                       - np.array([self._diff_value(data, classes, k, i, h_idx) * sigm[i][h_idx]
                                   for h_idx in hit]).sum()
               ) / (np.array([abs(sigm[i][m_idx]) for c in miss.keys() for m_idx in miss[c]]).sum() +
                    np.array([abs(sigm[i][h_idx]) for h_idx in hit]).sum())

    def sigmoid_matrix(self, dists: np.ndarray, t: float, u: float) -> np.ndarray:
        exponent_part = 1 + np.exp(-(t - dists) / (u / 4))

        return (2 / exponent_part) - 1

    def set_u(self, u: float):
        self.u = u

    def set_t(self, t: float):
        self.t = t

    def fill_matrix(self, d: np.ndarray) -> np.ndarray:
        for i in range(d.shape[0]):
            for j in range(i + 1, d.shape[0]):
                d[j][i] = d[i][j]

        return d

    def dist_matrix(self, data: np.ndarray) -> np.ndarray:
        d = np.zeros((data.shape[0], data.shape[0]), dtype=float)
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                d[i][j] = np.abs(data[i] - data[j]).sum()

        return d
