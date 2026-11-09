import os
from copy import copy
import itertools as it
from logging import getLogger
from functools import cached_property
from contextlib import contextmanager
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite
import joblib

from . import arr_utils

log = getLogger('sneqtl')


class _iLocIndexer:
    def __init__(self, parent, frozen=True):
        self.parent = parent
        self.frozen = frozen

    def __getitem__(self, key):
        return self.parent._data[key]

    def __setitem__(self, key, value):
        if frozen:
            raise KeyError(f'cannot set item on frozen object {self.parent.name}')
        self.parent._data[key] = value


class _LocIndexer:
    def __init__(self, parent, frozen=True):
        self.parent = parent
        self.frozen = frozen
        self.mappings = {
            0: parent._index_map,
            1: parent._column_map,
            2: parent._item_map,
        }
        self.ndim = parent._data.ndim
        self.shape = parent._data.shape
        

    def _key_to_idx(self, key, axis):
        mapping = self.mappings[axis]
        if isinstance(key, (list, np.ndarray)):
            return [mapping[k] for k in key]
        elif key == slice(None) or key is Ellipsis:
            return slice(None)
        else:
            return mapping[key]

    def _resolve_idx(self, key):
        if isinstance(key, tuple):
            if len(key) == 2:
                row_key, col_key = key
                item_idx = slice(None)
            elif len(key) == 3:
                if self.ndim < 3:
                    raise IndexError('Too many indices to array')
                row_key, col_key, item_key = key
                item_idx = self._key_to_idx(item_key, 2)
            else:
                raise IndexError('Too many indices to array')

            row_idx = self._key_to_idx(row_key, 0)
            col_idx = self._key_to_idx(col_key, 1)
            if isinstance(row_idx, list) and isinstance(col_idx, list):
                row_idx, col_idx = np.ix_(row_idx, col_idx)
        else:
            row_idx = self._key_to_idx(key, 0)
            col_idx = slice(None)
            item_idx = slice(None)
        return row_idx, col_idx, item_idx

    def __getitem__(self, key):
        row_idx, col_idx, item_idx = self._resolve_idx(key)
        if self.parent._data.ndim == 2:
            return self.parent._data[row_idx, col_idx]
        else:
            return self.parent._data[row_idx, col_idx][..., item_idx]

    def __setitem__(self, key, value):
        if self.frozen:
            raise KeyError(f'cannot set item on frozen object {self.parent.name}')
        row_idx, col_idx, item_idx = self._resolve_idx(key)
        if self.parent._data.ndim == 2:
            self.parent._data[row_idx, col_idx] = value
        else:
            self.parent._data[row_idx, col_idx][..., item_idx] = value


class VariableManager:

    index_name = None
    columns_name = None
    
    def __init__(self, data, index, columns, items=None, name=None, dtype=None, frozen=True):
        self.name = self.__class__.__name__ if name is None else name
        if dtype is not None:
            data = data.astype(dtype, copy=False)
        self._data = data
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
        self.iloc = _iLocIndexer(self, frozen)
        self.loc = _LocIndexer(self, frozen)
        self.frozen = bool(frozen)

    @property
    def is_memmap(self):
        return isinstance(self._data, np.memmap)

    def copy(self):
        return self.__class__(
            np.array(self._data, copy=True),
            index=self.index,
            columns=self.columns,
            items=self.items,
            name=copy(self.name),
            dtype=self._data.dtype
        )

    @cached_property
    def index(self):
        return list(self._index_map.keys())

    @cached_property
    def columns(self):
        return list(self._column_map.keys())

    @cached_property
    def items(self):
        return list(self._item_map.keys()) if self._item_map else None

    def __getitem__(self, key):
        if isinstance(key, list):
            idxs = [self._column_map[k] for k in key]
            return self._data[:, idxs, ...]
        return self._data[:, self._column_map[key], ...]

    def __setitem__(self, key, value):
        if self.frozen:
            raise KeyError(f'cannot set item on frozen object {self.name}')
        self._data[:, self._column_map[key], ...] = value

    def __len__(self):
        return len(self._index_map)

    def rename(self, name):
        self.name = name

    @property
    def values(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    def freeze(self):
        self.frozen = True

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

    def drop(self, index=None, columns=None, items=None):
        index = [i for i in self.index if i not in index] if index is not None else None
        columns = [i for i in self.columns if i not in columns] if columns is not None else None
        items = [i for i in self.items if i not in items] if items is not None else None
        return self.reindex(index=index, columns=columns, items=items)

    def dropna(self, axis=0, how='any'):
        if how not in ('any', 'all'):
            raise ValueError("how must be 'any' or 'all'")
        reduce_axes = [i for i in range(self.ndim) if i != axis]
        reduce_method = np.any if how == 'any' else np.all
        drop_mask = reduce_method(np.isnan(self._data), axis=reduce_axes)
        axis_name = {0: 'index', 1: 'columns', 2: 'items'}[axis]
        axis_labels = getattr(self, axis_name)
        keep_labels = [lab for lab, dm in zip(axis_labels, drop_mask) if not dm]
        return self.reindex(**{axis_name: keep_labels})

    @classmethod
    def from_dataframe(cls, df, copy=True):
        data = df.to_numpy(copy=copy)
        return cls(
            data=data,
            index=df.index.tolist(),
            columns=df.columns.tolist(),
            dtype=data.dtype,
        )

    def to_dataframe(self, copy=True):
        if self._data.ndim == 2:
            return pd.DataFrame(
                self._data,
                index=pd.Index(self.index, name=self.index_name),
                columns=pd.Index(self.columns, name=self.columns_name),
                copy=copy
            )
        elif self._data.ndim == 3:
            raise NotImplementedError('Cannot convert 3D variable manager to dataframe')

    def as_numeric(self, axis=0, drop_first=True, dtype=np.float32):
        if axis not in (0, 1):
            raise ValueError('axis must be 0 or 1')

        if np.issubdtype(self._data.dtype, np.number):
            return self

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
            dummies, categories = arr_utils.get_dummies(col, drop_first=drop_first, dtype=dtype)
            dummy_list.append(dummies)
            columns += [f'{col_name}_{cat}' for cat in categories]

        data = np.concatenate(dummy_list, axis=1)

        if axis == 1:
            data = data.T
            columns = index
            index = columns

        return VariableManager(data, index=index, columns=columns)

    @property
    def has_row_sum_constraint(self):
        if self.ndim != 2:
            raise ValueError('has_row_sum_constraint is only valid for 2D arrays')

        tol = 1e-10
        X = self._data
        is_constant = np.all(X == X[0, :], axis=0)
        X = X[:, ~is_constant]

        n_cols = X.shape[1]
        if not n_cols:
            return True

        # test row-sum constraint
        row_sums = X.sum(axis=1)
        return np.all(np.abs(row_sums - row_sums[0]) < tol)

    @property
    def is_singular(self):
        if self.ndim != 2:
            raise ValueError('is_singular is only valid for 2D arrays')

        tol = 1e-10
        X = self._data
        is_constant = np.all(X == X[0, :], axis=0)
        X = X[:, ~is_constant]

        n_cols = X.shape[1]
        if not n_cols:
            return True

        rank = np.linalg.matrix_rank(X, tol=tol)
        return rank < n_cols

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

    def drop_rank_dependent(self, tol=1e-8):
        if self.ndim != 2:
            raise ValueError("X must be 2D")
        n, n_feat = self._data.shape
        keep_columns = self.columns
        X_i = np.concatenate([np.ones((n, 1)), self._data.astype(float)], axis=1)
        while len(keep_columns) > 1:
            if np.linalg.matrix_rank(X_i, tol=tol) == (n_feat + 1):
                break
            U, s, Vt = np.linalg.svd(X_i, full_matrices=False)
            v = Vt[-1]
            j = int(np.argmax(np.abs(v[1:])))
            keep_columns = [col for i, col in enumerate(keep_columns) if i != j]
            X_i = np.delete(X_i, j + 1, axis=1)
            n_feat -= 1
        return self.reindex(columns=keep_columns)

    def as_covariates(self):
        covar = self.drop_uninformative(axis=1)
        covar = covar.drop_rank_dependent()
        return covar

    def get_correlated(self, other, min_r2):
        r2 = arr_utils.corrwith(other._data, self._data, axis=0) ** 2
        correlated = r2.max(axis=0) >= min_r2
        return self.reindex(
            columns=[col for k, col in zip(correlated, self.columns) if k]
        )

    def filter_correlated(self, other, max_r2):
        r2 = arr_utils.corrwith(other._data, self._data, axis=0) ** 2
        keep = np.nan_to_num(r2, nan=0.0).max(axis=0) <= max_r2
        return self.reindex(
            columns=[col for k, col in zip(keep, self.columns) if k]
        )


class SparseMixIn:

    sparse_mat_types = {
        'dok': sparse.dok_array,
        'csr': sparse.csr_array,
        'coo': sparse.coo_array,
        'lil': sparse.lil_array,
    }

    @property
    def is_sparse(self):
        return sparse.issparse(self._data)

    def as_sparse(self, sparse_type='csr'):
        try:
            sparse_mat_init = self.sparse_mat_types[sparse_type]
        except KeyError:
            raise ValueError('Sparse matrix type not recognised')
        if self.ndim == 3:
            raise ValueError('can only convert 2D objects to sparse format')
        return self.__class__(
            sparse_mat_init(self._data),
            index=self.index,
            columns=self.columns,
            items=None,
            name=self.name,
            frozen=self.frozen,
        )

    def as_dense(self):
        return self.__class__(
            self._data.toarray(),
            index=self.index,
            columns=self.columns,
            items=None,
            name=self.name,
            frozen=self.frozen,
        )

    @classmethod
    def empty_sparse(cls, index, columns, sparse_type='dok', dtype=None, name=None):
        try:
            sparse_mat_init = cls.sparse_mat_types[sparse_type]
        except KeyError:
            raise ValueError('Sparse matrix type not recognised')
        return cls(
            sparse_mat_init((len(index), len(columns)), dtype=dtype),
            index=index,
            columns=columns,
            name=name,
            frozen=False
        )

    @staticmethod
    def _read_axis(ax_fn):
        with open(ax_fn) as f:
            ax = [r.strip().split('\t')[0] for r in f.readlines()]
        return np.array(ax)

    @staticmethod
    def _write_axis(ax_fn, axis_labels):
        with open(ax_fn, 'w') as f:
            f.write('\n'.join(axis_labels))

    @classmethod
    def read_matrix_market(cls, mtx_dir, dtype=np.float32, name=None,
                           mtx_fn='matrix.mtx',
                           index_fn='index.tsv',
                           columns_fn='columns.tsv'):
        index = cls._read_axis(os.path.join(mtx_dir, index_fn))
        columns = cls._read_axis(os.path.join(mtx_dir, columns_fn))
        mtx = mmread(os.path.join(mtx_dir, mtx_fn)).astype(dtype, copy=False).tocsr()
        return cls(mtx, index=index, columns=columns, name=name)

    def write_matrix_market(self, output_dir,
                            mtx_fn='matrix.mtx',
                            index_fn='features.tsv',
                            columns_fn='barcodes.tsv'):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self._write_axis(os.path.join(output_dir, index_fn), self.index)
        self._write_axis(os.path.join(output_dir, columns_fn), self.columns)
        if self.is_sparse:
            data = self._data
        else:
            data = sparse.coo_array(self._data)
        mmwrite(os.path.join(output_dir, mtx_fn), data)