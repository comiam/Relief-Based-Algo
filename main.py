from IRelief import IRelief
from IterativeRelief import IterativeRelief
from MultiSURF import MultiSURF
from MultiSURFStar import MultiSURFStar
from SURF import SURF
from SURFStar import SURFStar
from SWRF import SWRF
from VLSReliefF import VLSReliefF
from data_utils.ClassDelemiter import split_classes
from data_utils.ColumnSearch import search_column_by_array
from data_utils.Parser import parse_first_csv
from RReliefF import RReliefF
from Relief import Relief
from ReliefF import ReliefF
from TuRF import TuRF

import numpy as np

from iVLSRelief import iVLSReliefF


def test_relief(data_t, classes_t):
    relief = Relief()
    for i in range(10):
        print(relief.fit(data_t, classes_t))


def test_swrf(data_t, classes_t):
    relief = SWRF(25, 4)

    return relief.fit(data_t, classes_t)


def test_multi_surf(data_t, classes_t):
    relief = MultiSURF(25)

    return relief.fit(data_t, classes_t)


def test_multi_surf_star(data_t, classes_t):
    relief = MultiSURFStar(25)

    return relief.fit(data_t, classes_t)


def test_relieff(data_t, classes_t):
    relief = ReliefF(100, 4)

    return np.array([relief.fit(data_t, classes_t) for i in range(10)]).mean(axis=0)


def test_surf(data_t, classes_t):
    relief = SURF(300)
    # print(relief.compute_avg_distance(data_t, classes_t))
    relief.set_threshold(3)

    return np.array([relief.fit(data_t, classes_t) for i in range(10)]).mean(axis=0)


def test_surfstar(data_t, classes_t):
    relief = SURFStar(300)
    print(relief.compute_avg_distance(data_t, classes_t))
    relief.set_threshold(relief.compute_avg_distance(data_t, classes_t) - 1)

    return np.array([relief.fit(data_t, classes_t) for i in range(10)]).mean(axis=0)


def test_vlsrelieff(data_t, classes_t):
    relief = VLSReliefF(100, 4)

    return np.array([relief.fit(data_t, classes_t, 18, 4) for i in range(3)]).mean(axis=0)


def test_ivlsrelieff(data_t, classes_t):
    relief = iVLSReliefF(6, 270, 4, 0.4)

    return relief.fit(data_t, classes_t, 20, 4)


def test_rrelieff(data_t, classes_t):
    unique, counts = np.unique(classes_t, return_counts=True)

    indexes = np.argsort(counts)
    print(classes_t, ' ', counts, ' ', counts[indexes[0]])

    relief = RReliefF(100, 5)
    return np.array([relief.fit(data_t, classes_t) for i in range(10)]).mean(axis=0)


def test_iterative_relief(data_t, classes_t):
    relief = IterativeRelief(kernel_width=50)
    return relief.fit(data_t, classes_t)[1]


def test_irelief(data_t, classes_t):
    relief = IRelief(reg_param=1, kernel_width=1, lr=0.005, stop_criterion=0.01)
    return relief.fit(data_t, classes_t)


def test_turf(data_t, classes_t, df):
    relief = TuRF(iterations=150, knn=4, turf_iteration_count=5, delete_features_per_iteration=3)

    res = relief.fit(data_t, classes_t)

    ind_res = np.argsort(res[1])
    for i in ind_res:
        print(search_column_by_array(res[0][:, i], df), ": ", res[1][i])


def relief_exec():
    data, classes, df = parse_first_csv("data/first.csv", True)

    classes = split_classes(classes)
    # test_relieff(data, classes)
    res = test_multi_surf_star(data, classes)
    # print(res)
    ind_res = np.argsort(res)

    for i in ind_res:
        print(df.columns[i], ": ", res.tolist()[i])

    # print("=======================")
    # test_relieff(data, classes)


if __name__ == '__main__':
    relief_exec()
