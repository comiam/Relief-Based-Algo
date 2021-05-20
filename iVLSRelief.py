import numpy as np

from VLSReliefF import VLSReliefF


# New parody of TuRF, but more powerful
# https://ieeexplore.ieee.org/document/4675767
class iVLSReliefF(VLSReliefF):
    def __init__(self, final_count_of_attrs: int, iterations: int = 100, knn: int = 10, percentile_rank: float = 0.5):
        super().__init__(iterations, knn)

        if not (0 < percentile_rank < 1):
            raise ValueError("invalid percentile rank!")

        if final_count_of_attrs <= 0:
            raise ValueError("invalid final count of attrs!")

        self.percentile_rank = percentile_rank
        self.final_count_of_attrs = final_count_of_attrs

    def fit(self, data: np.ndarray, classes: np.ndarray, subseq_count: int, subseq_length: int) -> np.ndarray:
        data_copy = data.copy()
        while True:
            w = super().fit(data_copy, classes, subseq_count, subseq_length)

            ind = np.argsort(w)
            if int(len(ind) * (1.0 - self.percentile_rank)) <= self.final_count_of_attrs:
                total_deletions = len(ind) - self.final_count_of_attrs
            else:
                total_deletions = int(len(ind) * self.percentile_rank)

            print(total_deletions)

            data_copy = np.delete(data_copy, ind[-total_deletions:], axis=1)
            w = np.delete(w, ind[-total_deletions:])

            if data_copy.shape[1] <= self.final_count_of_attrs:
                return w
