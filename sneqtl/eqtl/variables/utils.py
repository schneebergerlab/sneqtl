import numpy as np
from .base import VariableManager


def is_numeric(arr):
    try:
        arr.astype(float)
        return True
    except (ValueError, TypeError):
        return False


def get_dummies(arr, categories=None, drop_first=True, dtype=np.float32):
    if categories is None:
        categories = np.unique(arr)
    if drop_first:
        categories = categories[1:]
    dummies = (arr[:, None] == categories).astype(dtype)
    return dummies, categories


def get_numeric_variables(variables, axis=0, drop_first=True, dtype=np.float32):
    if axis not in (0, 1):
        raise ValueError('axis must be 0 or 1')

    if axis == 1:
        variables = variables.T

    n_samples, n_features = variables.shape
    index = variables.index
    dummy_list = []
    columns = []

    for i in range(n_features):
        col = variables.iloc[:, i]
        col_name = variables.columns[i]
        if is_numeric(col):
            col = col.reshape(-1, 1).astype(dtype)
            dummy_list.append(col)
            columns.append(col_name)
        else:
            dummies, categories = get_dummies(col, drop_first=drop_first, dtype=dtype)
            dummy_list.append(dummies)
            columns += [f'{col_name}_{cat}' for cat in categories]

    data = np.concatenate(dummy_list, axis=1)

    if axis == 1:
        data = data.T
        columns = index
        index = columns

    return VariableManager(data, index=index, columns=columns)
