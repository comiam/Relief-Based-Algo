import pandas as pd
import numpy as np
from sklearn import preprocessing


def normalize_set(data: pd.DataFrame) -> pd.DataFrame:
    mms = preprocessing.MinMaxScaler()
    x_scaled = mms.fit_transform(data)

    return pd.DataFrame(x_scaled, columns=data.columns)


def move_column_to_back(data: pd.DataFrame, name: str) -> pd.DataFrame:
    return data[[col for col in data.columns if col != name] + [name]]


def perform_second(file: str):
    data = pd.read_csv(file, delimiter=',')

    data = move_column_to_back(data, "qo_lc[m3/d]")
    data = move_column_to_back(data, "qw_lc[m3/d]")
    data = move_column_to_back(data, "qg_lc[m3/d]")
    data.drop(data.columns[0], axis=1, inplace=True)

    data.to_csv("data/mod5.csv", sep=',')


def parse_second_csv(file: str, test_class: int) -> tuple:
    data = pd.read_csv(file, delimiter=',')

    # qo_lc[m3/d]
    # qw_lc[m3/d]
    # qg_lc[m3/d]
    if test_class == 1:
        data = data.drop(data.columns[-2:], axis=1, inplace=False)
    elif test_class == 2:
        data = data.drop(data.columns[-1], axis=1, inplace=False)
        data = data.drop(data.columns[-2], axis=1, inplace=False)
    elif test_class == 2:
        data = data.drop(data.columns[-2], axis=1, inplace=False)
        data = data.drop(data.columns[-2], axis=1, inplace=False)

    data_features = data.drop(data.columns[-1], axis=1, inplace=False).to_numpy()
    classes = np.array(data.iloc[:, -1].values)

    return data_features, classes, data


def parse_first_csv(file: str, read_headers: bool) -> tuple:
    data = pd.read_csv(file, delimiter=',', header=None if not read_headers else "infer")

    data = normalize_set(data)

    data_features = data.drop(data.columns[-1], axis=1, inplace=False).to_numpy()
    classes = np.array(data.iloc[:, -1].values)

    return data_features, classes, data
