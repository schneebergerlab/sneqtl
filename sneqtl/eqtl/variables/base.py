import os
from logging import getLogger
from functools import cached_property
from contextlib import contextmanager
import tempfile

import numpy as np
import joblib

log = getLogger('sneqtl')


class _iLocIndexer:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        return self.parent._data[key]


class _LocIndexer:
    def __init__(self, parent):
        self.parent = parent

    @staticmethod
    def _to_idx(mapping, key):
        if key == slice(None):
            return slice(None)
        elif isinstance(key, list):
            return [mapping[k] for k in key]
        else:
            return mapping[key]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
            row_idx = self._to_idx(self.parent._index_map, row_key)
            col_idx = self._to_idx(self.parent._column_map, col_key)
            return self.parent._data[np.ix_(
                row_idx if isinstance(row_idx, list) else [row_idx],
                col_idx if isinstance(col_idx, list) else [col_idx]
            )]
        else:
            row_idx = self._to_idx(self.parent._index_map, key)
            return self.parent._data[row_idx]


class VariableManager:

    def __init__(self, data: np.ndarray, index=None, columns=None, dtype=np.float32):
        self._data = data.astype(dtype)
        if not len(set(index)) == len(index) or not len(index) == data.shape[0]:
            raise ValueError('Index values must be unique and match data length')
        if not len(set(columns)) == len(columns) or not len(columns) == data.shape[1]:
            raise ValueError('Column values must be unique and match data length')
        self._index_map = {k: i for i, k in enumerate(index)}
        self._column_map = {k: i for i, k in enumerate(columns)}

        self.iloc = _iLocIndexer(self)
        self.loc = _LocIndexer(self)

        self.is_memmap = isinstance(data, np.memmap)

    @classmethod
    def from_dataframe(cls, df, dtype=np.float32):
        return cls(
            df.to_numpy(),
            index=df.index.tolist(),
            columns=df.columns.tolist(),
        )

    @cached_property
    def index(self):
        return tuple(self._index_map.keys())

    @cached_property
    def columns(self):
        return tuple(self._column_map.keys())

    def __getitem__(self, key):
        if isinstance(key, list):
            idxs = [self._column_map[k] for k in key]
            return self._data[:, idxs]
        return self._data[:, self._column_map[key]]

    def reindex(self, index=None, columns=None):
        if self.is_memmap:
            log.warning('reindexing a memmapped VariableManager will copy the data into memory')
        row_idx = slice(None)
        col_idx = slice(None)

        if index is not None:
            row_idx = [self._index_map[i] for i in index]

        if columns is not None:
            col_idx = [self._column_map[c] for c in columns]

        data = self._data[row_idx, col_idx]

        new_index = index if index is not None else list(self._index_map.keys())
        new_columns = columns if columns is not None else list(self._column_map.keys())

        return self.__class__(data, index=new_index, columns=new_columns)

    @property
    def shape(self):
        return self._data.shape

    def transpose(self):
        return self.__class__(
            self._data.T,
            index=self.columns,
            columns=self.index
        )

    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        return f'<{self.__class__.__name__} shape={self.shape} memmap={self.is_memmap}>'

    def memmap(self, path_prefix, load=False):
        if self.is_memmap:
            log.warning(f'{self} is already memmapped')
        log.debug(f'memmapping {self} to {path_prefix}.npy')
        data_path = path_prefix + '.npy'
        index_path = path_prefix + '_index.pickle'
        columns_path = path_prefix + '_columns.pickle'

        np.save(data_path, self._data)
        joblib.dump(list(self.index), index_path)
        joblib.dump(list(self.columns), columns_path)

        if load:
            return self.__class__.load_memmap(path_prefix)
        return self

    @classmethod
    def load_memmap(cls, path_prefix: str):
        data_path = path_prefix + '.npy'
        index_path = path_prefix + '_index.pickle'
        columns_path = path_prefix + '_columns.pickle'

        data = np.load(data_path, mmap_mode='r')
        index = joblib.load(index_path)
        columns = joblib.load(columns_path)

        return cls(data, index=index, columns=columns)

    @contextmanager
    def memmap_context(self, path_prefix=None, load=False):
        if path_prefix is None:
            tmpdir = tempfile.mkdtemp()
            path_prefix = os.path.join(tmpdir, self.__class__.__name__)
            cleanup_tmpdir = True
        else:
            cleanup_tmpdir = False
        try:
            mm = self.memmap(path_prefix, load=load)
            yield mm
        finally:
            if mm.is_memmap:
                del mm._data
            for suffix in ['.npy', '_index.pickle', '_columns.pickle']:
                try:
                    os.remove(path_prefix + suffix)
                except FileNotFoundError:
                    pass
            if cleanup_tmpdir:
                try:
                    os.rmdir(tmpdir)
                except OSError:
                    pass
