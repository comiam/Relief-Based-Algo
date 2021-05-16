import pandas as pd
import numpy as np


def parse_csv(file: str) -> tuple:
    data = pd.read_csv(file, delimiter=',', header=None)

    data_features = data.drop(data.columns[-1], axis=1, inplace=False).to_numpy()
    classes = np.array(data.iloc[:, -1].values)

    return data_features, classes



