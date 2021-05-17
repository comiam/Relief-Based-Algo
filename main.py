from IRelief import IRelief
from IterativeRelief import IterativeRelief
from Parser import parse_csv
from RReliefF import RReliefF
from Relief import Relief
from ReliefF import ReliefF
from TuRF import TuRF


def test_relief(data_t, classes_t):
    relief = Relief()
    for i in range(10):
        print(relief.fit(data_t, classes_t))


def test_relieff(data_t, classes_t):
    relief = ReliefF(40, 2)
    for i in range(10):
        print(relief.fit(data_t, classes_t))


def test_rrelieff(data_t, classes_t):
    relief = RReliefF(40, 4)
    for i in range(10):
        print(relief.fit(data_t, classes_t))


def test_iterative_relief(data_t, classes_t):
    relief = IterativeRelief()
    for i in range(10):
        res = relief.fit(data_t, classes_t)
        print(res[0], ' ', res[1])


def test_irelief(data_t, classes_t):
    relief = IRelief()
    for i in range(10):
        print(relief.fit(data_t, classes_t).tolist())


def test_turf(data_t, classes_t):
    relief = TuRF(iterations=16, knn=2, turf_iteration_count=3, delete_features_per_iteration=1)
    for i in range(10):
        res = relief.fit(data_t, classes_t)
        print(res[0])
        print("-------------------")
        print(res[1])
        print("=========================================================")


if __name__ == '__main__':
    data, classes = parse_csv("data/test-relieff-data.csv")
    test_irelief(data, classes)
    print("=======================")
    test_relieff(data, classes)
