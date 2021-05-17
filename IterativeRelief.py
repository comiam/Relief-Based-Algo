import numpy as np
from numpy import linalg as la


#
# Some code taken from mlpy 2.2.0
#

class IterativeRelief:
    def __init__(self, data: np.ndarray, classes: np.ndarray, iteration_count: int = 100, sigma: float = 1.0,
                 stop_criteria: float = 0.001):
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if sigma <= 0:
            raise ValueError("Invalid sigma value!")

        if iteration_count <= 0:
            raise ValueError("Invalid iteration count!")

        if stop_criteria <= 0:
            raise ValueError("Invalid stop criteria count")

        self.sigma = sigma
        self.iteration_count = iteration_count
        self.stop_criteria = stop_criteria
        self.data = data
        self.classes = classes

    def fit(self) -> tuple[int, np.ndarray]:
        it = 0
        w_old = np.array([1 / self.data.shape[1]] * self.data.shape[1])

        misses, hits = self.find_misses_hits()
        for i in range(self.iteration_count):
            it += 1
            w = self.compute_w(w_old, misses, hits)

            w_old = w
            if la.norm(w - w_old, 2) < self.stop_criteria:
                break

        return it + 1, w_old

    def compute_w(self, old_w: np.ndarray, misses: list, hits: list):
        I = self.data.shape[1]

        ni = np.zeros(I, dtype=float)
        dist_k = self.compute_kernel_ds(old_w)
        for n in range(self.data.shape[0]):
            m_n = np.zeros(I, dtype=float)
            h_n = np.zeros(I, dtype=float)
            for i in misses[n]:
                a_in = self.compute_prob(dist_k, i, n, misses[n])
                m_in = abs(self.data[n] - self.data[i])
                m_n += a_in * m_in
            for i in hits[n]:
                b_in = self.compute_prob(dist_k, i, n, hits[n])
                h_in = abs(self.data[n] - self.data[i])
                h_n += b_in * h_in
            g_n = self.compute_outlier_prob(dist_k, n, misses[n])
            ni += g_n * (m_n - h_n)

        ni /= self.data.shape[0]

        ni_p = np.maximum(ni, 0.0)
        ni_p_norm2 = la.norm(ni_p, 2)

        return ni_p / ni_p_norm2

    def find_misses_hits(self) -> tuple[list, list]:
        m, h = [], []
        for n in range(self.classes.shape[0]):
            m_n = np.where(self.classes != self.classes[n])[0].tolist()
            m.append(m_n)
            h_n = np.where(self.classes == self.classes[n])[0]
            h_n = h_n[h_n != n].tolist()
            h.append(h_n)

        return m, h

    def compute_kernel_ds(self, w: np.ndarray) -> np.ndarray:
        d = np.zeros((self.data.shape[0], self.data.shape[0]), dtype=float)
        for i in range(self.data.shape[0]):
            for j in range(i + 1, self.data.shape[0]):
                d[i][j] = self.w_norm(self.data[i] - self.data[j], w)
                d[j][i] = d[i][j]
        dk = np.exp(-d / self.sigma)

        return dk

    def compute_prob(self, dist_k: np.ndarray, i: int, n: int, indices: list) -> float:
        den = dist_k[n][indices].sum()
        if den == 0.0:
            raise Exception("sigma (kernel parameter) too small")

        return dist_k[n][i] / den

    def compute_outlier_prob(self, dist_k, n, misses_n):
        num = dist_k[n][misses_n].sum()
        index_range = [i for i in range(self.data.shape[0])]
        index_range.remove(n)
        den = dist_k[n][index_range].sum()
        if den == 0.0:
            raise Exception("sigma (kernel parameter) too small")

        return 1.0 - (num / den)

    def w_norm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return (w * abs(x)).sum()
