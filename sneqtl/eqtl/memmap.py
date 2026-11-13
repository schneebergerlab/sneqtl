import os
import logging
import shutil
import joblib
import tempfile

import numpy as np

from sneqtl.variables.base import VariableManager

log = logging.getLogger('sneqtl')


class MemmapManager:

    def __init__(self, cache=None, persist=False):
        if persist and cache is None:
            raise ValueError('persist=True requires a cache')

        self.cache = cache
        self.persist = persist
        self.cached = cache is not None and cache.is_complete()
        self.is_tmp = not self.cached and not persist

        if self.cached or persist:
            self.memmap_dir = cache.path
        else:
            self.memmap_dir = tempfile.mkdtemp()

        self.memmap_prefixes = {}
        self.memmap_classes = {}

        if self.cached:
            for name, cls in cache.datasets.items():
                self.register(name, cls)

    def register(self, name, cls):
        path_prefix = os.path.join(self.memmap_dir, name)
        self.memmap_prefixes[name] = path_prefix
        self.memmap_classes[name] = cls
        return path_prefix

    def convert(self, dataset):
        if self.cached:
            raise ValueError('Cannot overwrite a cached dataset')

        dataset_name = dataset.name
        dataset_cls = dataset.__class__
        path_prefix = self.register(dataset_name, dataset_cls)
        data_path = path_prefix + '.npy'
        index_path = path_prefix + '_index.pickle'
        columns_path = path_prefix + '_columns.pickle'
        items_path = path_prefix + '_items.pickle'

        if not getattr(dataset, 'is_sparse', False):
            np.save(data_path, dataset._data)
        else:
            data = np.lib.format.open_memmap(
                data_path, mode='w+', dtype=dataset._data.dtype, shape=dataset.shape,
            )
            for i in range(0, dataset.shape[0]):
                data[i, :] = dataset._data[i, :].toarray()
            data.flush()
            del data
        joblib.dump(list(dataset.index), index_path)
        joblib.dump(list(dataset.columns), columns_path)
        if dataset._item_map:
            joblib.dump(list(dataset.items), items_path)

        attrs = {
            attr: getattr(dataset, attr)
            for attr in dataset.memmap_attrs
        }
        if attrs:
            joblib.dump(attrs, path_prefix + '_attrs.pickle')

        return self.load(dataset_name)

    def load(self, dataset_name, mode='r', path_prefix=None, cls=None):
        if self.cached and mode != 'r':
            raise ValueError('Cached datasets are read-only')

        if path_prefix is None:
            path_prefix = self.memmap_prefixes[dataset_name]
        if cls is None:
            cls = self.memmap_classes[dataset_name]
        data_path = path_prefix + '.npy'
        index_path = path_prefix + '_index.pickle'
        columns_path = path_prefix + '_columns.pickle'
        items_path = path_prefix + '_items.pickle'

        data = np.load(data_path, mmap_mode=mode)
        index = joblib.load(index_path)
        columns = joblib.load(columns_path)
        items = joblib.load(items_path) if os.path.exists(items_path) else None

        mm = cls(data, index=index, columns=columns, items=items, name=dataset_name)
        if cls.memmap_attrs:
            attrs = joblib.load(path_prefix + '_attrs.pickle')
            for attr, value in attrs.items():
                setattr(mm, attr, value)
        mm._memmap_prefix = path_prefix
        return mm

    def create_empty(self, dataset_name, index, columns, items=None,
                     dtype=np.float32, cls=VariableManager):
        if self.cached:
            raise ValueError('Cannot overwrite a cached dataset')

        if dataset_name is None:
            dataset_name = cls.__name__
        if items is None:
            shape = (len(index), len(columns))
        else:
            shape = (len(index), len(columns), len(items))
        path_prefix = self.register(dataset_name, cls)

        data_path = path_prefix + '.npy'
        index_path = path_prefix + '_index.pickle'
        columns_path = path_prefix + '_columns.pickle'
        items_path = path_prefix + '_items.pickle'

        joblib.dump(list(index), index_path)
        joblib.dump(list(columns), columns_path)
        if items is not None:
            joblib.dump(list(items), items_path)
        data = np.lib.format.open_memmap(
            data_path, mode='w+', dtype=dtype, shape=shape
        )

        dataset = cls(data, index=index, columns=columns, items=items,
                      name=dataset_name, dtype=dtype, frozen=False)

        # although these are empty, they are required for some variables
        attrs = {
            attr: getattr(dataset, attr)
            for attr in dataset.memmap_attrs
        }
        if attrs:
            joblib.dump(attrs, path_prefix + '_attrs.pickle')
        dataset._memmap_prefix = path_prefix
        return dataset

    def remove_all(self):
        if self.is_tmp and os.path.isdir(self.memmap_dir):
            shutil.rmtree(self.memmap_dir)
        self.memmap_prefixes = {}
        self.memmap_classes = {}
