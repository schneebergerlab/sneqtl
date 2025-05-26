import os
import logging
import shutil
import joblib
import tempfile

import numpy as np

from .variables.base import VariableManager 

log = logging.getLogger('sneqtl')


class MemmapManager:

    def __init__(self, memmap=True, memmap_dir=None):
        self.memmap = memmap
        if memmap_dir:
            self.memmap_dir = memmap_dir
            self.is_tmp = False
        elif memmap:
            self.memmap_dir = tempfile.mkdtemp()
            self.is_tmp = True
        else:
            self.memmap_dir = None
            self.is_tmp = False
        self.memmap_prefixes = {}
        self.memmap_classes = {}

    def register(self, name, cls):
        path_prefix = os.path.join(self.memmap_dir, name)
        self.memmap_prefixes[name] = path_prefix
        self.memmap_classes[name] = cls
        return path_prefix

    def convert(self, dataset):
        if not self.memmap:
            return dataset
        dataset_name = dataset.name
        dataset_cls = dataset.__class__
        path_prefix = self.register(dataset_name, dataset_cls)
        data_path = path_prefix + '.npy'
        index_path = path_prefix + '_index.pickle'
        columns_path = path_prefix + '_columns.pickle'
        items_path = path_prefix + '_items.pickle'

        np.save(data_path, dataset._data)
        joblib.dump(list(dataset.index), index_path)
        joblib.dump(list(dataset.columns), columns_path)
        if dataset._item_map:
            joblib.dump(list(dataset.items), items_path)

        return self.load(dataset_name)

    def load(self, dataset_name, mode='r', path_prefix=None, cls=None):
        if not self.memmap:
            raise ValueError('dataset is not memmapped')
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
        mm._memmap_prefix = path_prefix
        return mm

    def create_empty(self, dataset_name, index, columns, items=None,
                     dtype=np.float32, cls=VariableManager):
        if dataset_name is None:
            dataset_name = cls.__name__
        if items is None:
            shape = (len(index), len(columns))
        else:
            shape = (len(index), len(columns), len(items))
        path_prefix = self.register(dataset_name, cls)

        if self.memmap:
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
        else:
            data = np.empty(shape=shape, dtype=dtype)
        dataset = cls(data, index=index, columns=columns, items=items,
                      name=dataset_name, dtype=dtype)
        dataset._memmap_prefix = path_prefix
        return dataset

    def remove_all(self):
        if self.is_tmp and os.path.isdir(self.memmap_dir):
            shutil.rmtree(self.memmap_dir)
        self.memmap_prefixes = {}
        self.memmap_classes = {}
