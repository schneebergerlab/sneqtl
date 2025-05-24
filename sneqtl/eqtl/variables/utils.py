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


def corrwith(arr1, arr2, axis=0):
    if axis not in (0, 1):
        raise ValueError('axis must be 0 or 1')

    if axis == 1:
        arr1, arr2 = arr1.T, arr2.T

    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError('Input arrays must have the same size along the correlation axis')

    arr1_centered = arr1 - arr1.mean(axis=0, keepdims=True)
    arr2_centered = arr2 - arr2.mean(axis=0, keepdims=True)

    num = np.einsum('ij,ik->jk', arr1_centered, arr2_centered)
    denom = np.sqrt(
        np.sum(arr1_centered**2, axis=0)[:, None] * np.sum(arr2_centered**2, axis=0)[None, :]
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        r = (num / denom)**2
        r[denom == 0] = np.nan

    return r