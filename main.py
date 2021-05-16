from Parser import parse_csv
from RReliefF import RReliefF
from Relief import Relief
from ReliefF import ReliefF
from TuRF import TuRF


def test_relief(data_t, classes_t):
    relief = Relief(data_t, classes_t, 10)
    for i in range(10):
        print(relief.fit())


def test_relieff(data_t, classes_t):
    relief = ReliefF(data_t, classes_t, 40, 2)
    for i in range(10):
        print(relief.fit())


def test_rrelieff(data_t, classes_t):
    relief = RReliefF(data_t, classes_t, 40, 4)
    for i in range(10):
        print(relief.fit())


def test_turf(data_t, classes_t):
    for i in range(10):
        relief = TuRF(data_t, classes_t, iterations=16, knn=2, turf_iteration_count=3, delete_features_per_iteration=1)
        res = relief.fit()
        print(res[0])
        print("-------------------")
        print(res[1])
        print("=========================================================")


if __name__ == '__main__':
    data, classes = parse_csv("data/test-turf-data.csv")
    test_turf(data, classes)
