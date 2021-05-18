from IRelief import IRelief
from IterativeRelief import IterativeRelief
from data_utils.ClassDelemiter import split_classes
from data_utils.ColumnSearch import search_column_by_array
from data_utils.Parser import parse_csv
from RReliefF import RReliefF
from Relief import Relief
from ReliefF import ReliefF
from TuRF import TuRF

import numpy as np


def test_relief(data_t, classes_t):
    relief = Relief()
    for i in range(10):
        print(relief.fit(data_t, classes_t))


def test_relieff(data_t, classes_t):
    relief = ReliefF(40, 2)

    return np.array([relief.fit(data_t, classes_t) for i in range(10)]).mean(axis=0)


def test_rrelieff(data_t, classes_t):
    relief = RReliefF(100, 4)
    return np.array([relief.fit(data_t, classes_t) for i in range(10)]).mean(axis=0)


def test_iterative_relief(data_t, classes_t):
    relief = IterativeRelief(kernel_width=50)
    return relief.fit(data_t, classes_t)[1]


def test_irelief(data_t, classes_t):
    relief = IRelief(kernel_width=7)
    return relief.fit(data_t, classes_t)


def test_turf(data_t, classes_t, df):
    relief = TuRF(iterations=100, knn=2, turf_iteration_count=15, delete_features_per_iteration=1)

    res = relief.fit(data_t, classes_t)

    ind_res = np.argsort(res[1])
    for i in ind_res:
        print(search_column_by_array(res[0][:, i], df), ": ", res[1][i])


if __name__ == '__main__':
    data, classes, df = parse_csv("data/cleared.csv", True)

    classes = split_classes(classes)
    # test_turf(data, classes, df)
    res = test_irelief(data, classes)
    # print(res)
    ind_res = np.argsort(res)

    for i in ind_res:
        print(df.columns[i], ": ", res.tolist()[i])

    print("=======================")
    test_relieff(data, classes)
