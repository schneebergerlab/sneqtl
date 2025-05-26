import os
from logging import getLogger
from functools import cached_property
from contextlib import contextmanager
import tempfile

import numpy as np
import joblib

from . import utils

log = getLogger('sneqtl')


class _iLocIndexer:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        return self.parent._data[key]


class _LocIndexer:
    def __init__(self, parent):
        self.parent = parent
        self.ndim = parent._data.ndim
        self.shape = parent._data.shape

    def _to_idx(self, mapping, key, dim):
        if isinstance(key, (list, np.ndarray)):
            return [mapping[k] for k in key]
        elif key == slice(None):
            return np.arange(self.shape[dim])
        else:
            return mapping[key]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 2:
                row_key, col_key = key
                item_idx = slice(None)
            elif len(key) == 3:
                if self.ndim < 3:
                    raise IndexError('Too many indices to array')
                row_key, col_key, item_key = key
                item_idx = self._to_idx(self.parent._item_map, item_key, 2)
            else:
                raise IndexError('Too many indices to array')

            row_idx = self._to_idx(self.parent._index_map, row_key, 0)
            col_idx = self._to_idx(self.parent._column_map, col_key, 1)

            if self.parent._data.ndim == 2:
                return self.parent._data[np.ix_(row_idx, col_idx)]
            else:
                return self.parent._data[np.ix_(row_idx, col_idx)][..., item_idx]
        else:
            row_idx = self._to_idx(self.parent._index_map, key, 0)
            return self.parent._data[row_idx]


class VariableManager:
    def __init__(self, data, index, columns, items=None, name=None, dtype=np.float32):
        self.name = self.__class__.__name__ if name is None else name
        self._data = data.astype(dtype, copy=False)
        if data.ndim not in (2, 3):
            raise ValueError('data must be 2D or 3D (rows x cols x items)')

        if data.ndim == 2:
            n_rows, n_cols = data.shape
            n_items = None
        else:
            n_rows, n_cols, n_items = data.shape

        if len(set(index)) != len(index) or len(index) != n_rows:
            raise ValueError('Index values must be unique and match data rows')
        if len(set(columns)) != len(columns) or len(columns) != n_cols:
            raise ValueError('Column values must be unique and match data columns')

        self._index_map = {k: i for i, k in enumerate(index)}
        self._column_map = {k: i for i, k in enumerate(columns)}
        self._item_map = None
        if items is not None:
            if data.ndim != 3:
                raise ValueError('items provided but data is not 3D')
            if len(set(items)) != len(items) or len(items) != n_items:
                raise ValueError('Item values must be unique and match data items')
            self._item_map = {k: i for i, k in enumerate(items)}

        self._memmap_prefix = None
        self.iloc = _iLocIndexer(self)
        self.loc = _LocIndexer(self)

    @property
    def is_memmap(self):
        return isinstance(self._data, np.memmap)

    @classmethod
    def from_dataframe(cls, df, dtype=np.float32):
        return cls(
            df.to_numpy(copy=False, dtype=dtype),
            index=df.index.tolist(),
            columns=df.columns.tolist(),
            dtype=dtype,
        )

    @cached_property
    def index(self):
        return tuple(self._index_map.keys())

    @cached_property
    def columns(self):
        return tuple(self._column_map.keys())

    @cached_property
    def items(self):
        return tuple(self._item_map.keys()) if self._item_map else None

    def __getitem__(self, key):
        if isinstance(key, list):
            idxs = [self._column_map[k] for k in key]
            return self._data[:, idxs, ...]
        return self._data[:, self._column_map[key], ...]

    def reindex(self, index=None, columns=None, items=None):
        index = index if index is not None else slice(None)
        columns = columns if columns is not None else slice(None)
        items = items if items is not None else slice(None)

        if self._data.ndim == 3:
            data = self.loc[index, columns, items]
        else:
            data = self.loc[index, columns]

        return self.__class__(
            data,
            index=index if index != slice(None) else self.index,
            columns=columns if columns != slice(None) else self.columns,
            items=items if items != slice(None) else self.items,
            name=self.name,
            dtype=data.dtype
        )

    def as_numeric(self, axis=0, drop_first=True, dtype=np.float32):
        if axis not in (0, 1):
            raise ValueError('axis must be 0 or 1')

        variables = self
        if axis == 1:
            variables = variables.T

        n_samples, n_features = variables.shape
        index = variables.index
        dummy_list = []
        columns = []

        for i in range(n_features):
            col = variables.iloc[:, i]
            col_name = variables.columns[i]
            if utils.is_numeric(col):
                col = col.reshape(-1, 1).astype(dtype)
                dummy_list.append(col)
                columns.append(col_name)
            else:
                dummies, categories = utils.get_dummies(col, drop_first=drop_first, dtype=dtype)
                dummy_list.append(dummies)
                columns += [f'{col_name}_{cat}' for cat in categories]

        data = np.concatenate(dummy_list, axis=1)

        if axis == 1:
            data = data.T
            columns = index
            index = columns

        return VariableManager(data, index=index, columns=columns)

    def as_covariates(self):
        return self

    def drop_uninformative(self, tol=1e-8, axis=0):
        if self._data.ndim == 2:
            axis_map = {0: self.index, 1: self.columns}
            axis_reduce = {0: 1, 1: 0}
        elif self._data.ndim == 3:
            axis_map = {0: self.index, 1: self.columns, 2: self.items}
            axis_reduce = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        else:
            raise ValueError('Unsupported data dimensionality')

        axis_label = axis_map[axis]
        variances = np.var(self._data, axis=axis_reduce[axis])

        mask = variances > tol
        axis_values = [k for k, keep in zip(axis_label, mask) if keep]

        kwargs = dict(index=None, columns=None, items=None)
        if axis == 0:
            kwargs['index'] = axis_values
        elif axis == 1:
            kwargs['columns'] = axis_values
        elif axis == 2:
            kwargs['items'] = axis_values
        return self.reindex(**kwargs)

    def get_correlated(self, other, min_r2):
        r2 = utils.corrwith(other._data, self._data, axis=0) ** 2
        correlated = r2.max(axis=0) >= min_r2
        return self.reindex(
            columns=[col for k, col in zip(correlated, self.columns) if k]
        )

    def filter_correlated(self, other, max_r2):
        r2 = utils.corrwith(other._data, self._data, axis=0) ** 2
        keep = r2.max(axis=0) <= max_r2
        return self.reindex(
            columns=[col for k, col in zip(keep, self.columns) if k]
        )

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    def transpose(self):
        if self._data.ndim == 2:
            data = self._data.T
        else:
            data = self._data.transpose(1, 0, 2)
        return self.__class__(
            data,
            index=self.columns,
            columns=self.index,
            items=self.items
        )

    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        dims = f'shape={self.shape}'
        return f'<{self.__class__.__name__} {dims} name={self.name} memmap={self.is_memmap}>'
