from operator import attrgetter
import itertools as it

import numpy as np


def _check_dimensions(*args):
    if len(args) < 1:
        raise ValueError('At least one object should be provided')
    ndim = args[0].ndim
    for arr in args[1:]:
        if arr.ndim != ndim:
            raise ValueError('Not all objects have the same number of dimensions')
    return ndim


def _check_aligned(args, attr):
    getter = attrgetter(attr)
    reference = getter(args[0])
    for arr in args[1:]:
        if arr is None:
            continue
        if getter(arr) != reference:
            raise ValueError(f'{attr} are not aligned across objects')
    return reference


def check_aligned(*args, axis=0):
    args = [a for a in args if a is not None]
    if len(args) < 1:
        raise ValueError('At least one object should be provided')
    attr = {0: 'index', 1: 'columns', 2: 'items'}[axis]
    return _check_aligned(args, attr)


def _add_suffix(axis, suffix):
    if suffix is None:
        return axis
    return (f'{str(v)}_{suffix}' for v in axis)


def concat(*args, axis=0, suffixes=None, name=None):
    if len(args) < 1:
        raise ValueError('At least one object should be provided')
    if suffixes is None:
        suffixes = [None,] * len(args)
    if len(suffixes) != len(args):
        raise ValueError('Length of suffixes must match number of input arrays')
    ndim = _check_dimensions(*args)
    attrs = ['index', 'columns', 'items'][:ndim]

    axes_labels = {}
    for i, attr in enumerate(attrs):
        if i == axis:
            values = it.chain.from_iterable(
                _add_suffix(getattr(a, attr), s) for (a, s) in zip(args, suffixes)
            )
            axes_labels[attr] = tuple(values)
        else:
            axes_labels[attr] = _check_aligned(args, attr)

    data = np.concatenate([v._data for v in args], axis=axis)
    cls = args[0].__class__
    return cls(
        data,
        name=name,
        **axes_labels
    )
