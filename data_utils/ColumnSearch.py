import numpy as np
import pandas as pd


def search_column_by_array(res: np.ndarray, df: pd.DataFrame) -> str:
    for column in df:
        compare = (np.equal(res, df[column].to_numpy(dtype=float)))
        if compare.all():
            return column

    return "empty"