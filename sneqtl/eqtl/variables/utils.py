from operator import attrgetter
import numpy as np


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


def check_aligned(*args, axis=0):
    if len(args) < 2:
        raise ValueError('At least two arrays should be provided')
    axisgetter = attrgetter('index') if axis == 0 else attrgetter('columns')
    reference_index = axisgetter(args[0])
    for arr in args[1:]:
        if arr is None:
            continue
        if axisgetter(arr) != reference_index:
            raise ValueError('Not all variable indexes are aligned')
    return reference_index


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