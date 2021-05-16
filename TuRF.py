from random import randrange

import numpy as np

from ReliefF import ReliefF


class TuRF(ReliefF):
    def __init__(self, data: np.ndarray, classes: np.ndarray, iterations: int, turf_iteration_count: int, knn: int = 10,
                 delete_features_per_iteration: int = 1):
        super().__init__(data, classes, iterations, knn)

        if turf_iteration_count * delete_features_per_iteration >= data.shape[1]:
            raise ValueError("The number of features to delete is greater than the original ones")

        self.turf_iteration_count = turf_iteration_count
        self.delete_features_per_iteration = delete_features_per_iteration
        self.data = data.copy()  # We don't have to edit the original dataset

    def fit(self) -> tuple[np.ndarray, np.ndarray]:
        w = None
        for i in range(self.turf_iteration_count):
            w = super(TuRF, self).fit()

            deleted_features = w.argsort()[0:self.delete_features_per_iteration]
            self.data = np.delete(self.data, deleted_features, axis=1)

            w = np.delete(w, deleted_features)

        return self.data, w
