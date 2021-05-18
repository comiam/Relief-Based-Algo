import numpy as np
from numpy import linalg as la


#
# Some code taken from mlpy 2.2.0
# IterativeRelief: one of the best upgrades of ReliefF https://ieeexplore.ieee.org/document/4160953

class IterativeRelief:
    def __init__(self, iteration_count: int = 100, kernel_width: float = 1.0,
                 stop_criterion: float = 0.001):
        if kernel_width <= 0:
            raise ValueError("Invalid kernel width value!")

        if iteration_count <= 0:
            raise ValueError("Invalid iteration count!")

        if stop_criterion <= 0:
            raise ValueError("Invalid stop criterion count!")

        self.kernel_width = kernel_width
        self.iteration_count = iteration_count
        self.stop_criterion = stop_criterion

    def fit(self, data: np.ndarray, classes: np.ndarray) -> tuple[int, np.ndarray]:
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        it = 0
        w_old = np.array([1.0 / data.shape[1]] * data.shape[1], dtype=float)

        misses, hits = self._find_misses_hits(classes)
        for i in range(self.iteration_count):
            it += 1
            w = self._compute_w(data, w_old, misses, hits)

            if la.norm(w - w_old, 2) <= self.stop_criterion:
                w_old = w
                break

            w_old = w

        return it + 1, w_old

    def _compute_w(self, data: np.ndarray, old_w: np.ndarray, misses: list, hits: list):
        I = data.shape[1]

        v = np.zeros(I, dtype=float)
        dist_k = self._compute_kernel_ds(data, old_w)
        #print(dist_k)
        #input()
        for n in range(data.shape[0]):
            sm = self._compute_sample_margin(data, n, dist_k, hits, misses)
            g_n = self._compute_outlier_prob(data, dist_k, n, misses[n])
            v += g_n * sm

        v /= data.shape[0]

        ni_p = np.maximum(v, 0.0)
        ni_p_norm2 = la.norm(ni_p, 2)

        return ni_p / ni_p_norm2

    def _find_misses_hits(self, classes: np.ndarray) -> tuple[list, list]:
        m, h = [], []
        for n in range(classes.shape[0]):
            m_n = np.where(classes != classes[n])[0].tolist()
            m.append(m_n)
            h_n = np.where(classes == classes[n])[0]
            h_n = h_n[h_n != n].tolist()
            h.append(h_n)

        return m, h

    def _compute_kernel_ds(self, data: np.ndarray, w: np.ndarray) -> np.ndarray:
        d = np.zeros((data.shape[0], data.shape[0]), dtype=float)
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                d[i][j] = self._w_norm(data[i] - data[j], w)
                #print(d[i][j])
        dk = np.exp(-d / self.kernel_width)

        return dk

    def _compute_prob(self, dist_k: np.ndarray, i: int, n: int, indices: list) -> float:
        den = dist_k[n][indices].sum()
        if den == 0.0:
            raise Exception("sigma (kernel parameter) too small")

        return dist_k[n][i] / den

    def _compute_outlier_prob(self, data: np.ndarray, dist_k, n, misses_n):
        num = dist_k[n][misses_n].sum()
        index_range = [i for i in range(data.shape[0])]
        index_range.remove(n)
        den = dist_k[n][index_range].sum()
        if den == 0.0:
            raise Exception("sigma (kernel parameter) too small")

        return 1.0 - (num / den)

    def _w_norm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return (w * abs(x)).sum()

    def _compute_sample_margin(self, data: np.ndarray, n: int, dist_k: np.ndarray, hits: list,
                               misses: list) -> np.ndarray:
        I = data.shape[1]
        m_n = np.zeros(I, dtype=float)
        h_n = np.zeros(I, dtype=float)
        for i in misses[n]:
            a_in = self._compute_prob(dist_k, i, n, misses[n])
            m_in = abs(data[n] - data[i])
            m_n += a_in * m_in
        for i in hits[n]:
            b_in = self._compute_prob(dist_k, i, n, hits[n])
            h_in = abs(data[n] - data[i])
            h_n += b_in * h_in

        return m_n - h_n
