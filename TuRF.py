from random import randrange

import numpy as np

from ReliefF import ReliefF


# Simple upgrade of ReliefF
# https://link.springer.com/content/pdf/10.1007%2F978-3-540-71783-6_16.pdf
class TuRF(ReliefF):
    def __init__(self, iterations: int, turf_iteration_count: int, knn: int = 10,
                 delete_features_per_iteration: int = 1):
        super().__init__(iterations, knn)

        self.turf_iteration_count = turf_iteration_count
        self.delete_features_per_iteration = delete_features_per_iteration

    def fit(self, data: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.turf_iteration_count * self.delete_features_per_iteration >= data.shape[1]:
            raise ValueError("The number of features to delete is greater than the original ones")

        data_copy = data.copy()  # We don't have to edit the original dataset

        w = None
        for i in range(self.turf_iteration_count):
            w = super(TuRF, self).fit(data_copy, classes)

            deleted_features = w.argsort()[0:self.delete_features_per_iteration]
            data_copy = np.delete(data_copy, deleted_features, axis=1)

            w = np.delete(w, deleted_features)

        return data_copy, w
