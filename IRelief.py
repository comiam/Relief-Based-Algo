import numpy as np
from numpy import linalg as la

from IterativeRelief import IterativeRelief


# IRelief - evolution of iterative relief: https://ieeexplore.ieee.org/document/5342431
class IRelief(IterativeRelief):
    def __init__(self, lr: float = 0.07, kernel_width: float = 10.0, stop_criterion: float = 0.001,
                 reg_param: float = 1.0):
        super().__init__(1, kernel_width, stop_criterion)
        if kernel_width <= 0:
            raise ValueError("Kernel width less than zero!")
        if stop_criterion <= 0:
            raise ValueError("Stop criterion less than zero!")
        if reg_param <= 0:
            raise ValueError("Regularization parameter less than zero!")
        if lr <= 0:
            raise ValueError("Learning rate less than zero!")

        self.kernel_width = kernel_width
        self.stop_criterion = stop_criterion
        self.reg_param = reg_param
        self.lr = lr

    def fit(self, data: np.ndarray, classes: np.ndarray):
        if data.shape[0] != classes.shape[0]:
            raise ValueError("data and class shapes not equals!")

        if data.shape[0] == classes.shape[0] == 0:
            raise ValueError("zero data shapes!")

        I = data.shape[1]
        old_w = np.ones(I, dtype=float)
        old_v = np.ones(I, dtype=float)

        misses, hits = self._find_misses_hits(classes)

        ones = np.ones(I)

        while True:
            v = old_v - self.lr * ((self.reg_param * ones - self._compute_margin(data, hits, misses, old_w)) * old_v)
            w = v ** 2
            old_v = v

            if la.norm(w - old_w, 2) <= self.stop_criterion:
                old_w = w
                break
            old_w = w

        return old_w

    def _compute_margin(self, data: np.ndarray, hits: list, misses: list, old_w: np.ndarray) -> np.ndarray:
        dist_k = self._compute_kernel_ds(data, old_w)
        I = data.shape[1]
        margin = np.zeros(I)

        for n in range(data.shape[0]):
            z = self._compute_sample_margin(data, n, dist_k, hits, misses)
            power = -1 * (old_w * z).sum()
            exponent = 0.9999999999999999999 if np.abs(power) >= 1000 else np.exp(-1 * (old_w * z).sum())
            coeff = exponent / (1 + exponent)
            margin += coeff * z

        return margin
