import os
import logging
import shutil
import joblib
import tempfile

import numpy as np

log = logging.getLogger('sneqtl')


class MemmapManager:

    def __init__(self, memmap=True, memmap_dir=None):
        self.memmap = memmap
        if memmap_dir:
            self.memmap_dir = memmap_dir
            self.is_tmp = False
        else:
            self.memmap_dir = tempfile.mkdtemp()
            self.is_tmp = True
        self.memmap_prefixes = {}
        self.memmap_classes = {}

    def as_memmap(self, dataset):
        if dataset is None:
            return None
        dataset_name = dataset.name
        if dataset.is_memmap:
            self.memmap_prefixes[dataset_name] = dataset._memmap_prefix
            self.memmap_classes[dataset_name] = dataset.__class__
        elif self.memmap:
            prefix = os.path.join(self.memmap_dir, dataset_name)
            self.memmap_prefixes[dataset_name] = prefix
            self.memmap_classes[dataset_name] = dataset.__class__
            dataset = self.convert(dataset, prefix)
        return dataset

    def convert(self, dataset, path_prefix):
        dataset_name = dataset.name
        if dataset.is_memmap:
            log.warning(f'{self} is already memmapped')
        log.debug(f'memmapping {self} to {path_prefix}.npy')
        path_prefix = self.memmap_prefixes[dataset_name]
        data_path = path_prefix + '.npy'
        index_path = path_prefix + '_index.pickle'
        columns_path = path_prefix + '_columns.pickle'
        items_path = path_prefix + '_items.pickle'

        np.save(data_path, dataset._data)
        joblib.dump(list(dataset.index), index_path)
        joblib.dump(list(dataset.columns), columns_path)
        if dataset._item_map:
            joblib.dump(list(dataset.items), items_path)

        return self.load_memmap(dataset_name)

    def load_memmap(self, dataset_name):
        path_prefix = self.memmap_prefixes[dataset_name]
        cls = self.memmap_classes[dataset_name]
        data_path = path_prefix + '.npy'
        index_path = path_prefix + '_index.pickle'
        columns_path = path_prefix + '_columns.pickle'
        items_path = path_prefix + '_items.pickle'

        data = np.load(data_path, mmap_mode='r')
        index = joblib.load(index_path)
        columns = joblib.load(columns_path)
        items = joblib.load(items_path) if os.path.exists(items_path) else None

        mm = cls(data, index=index, columns=columns, items=items, name=dataset_name)
        mm._memmap_prefix = path_prefix
        return mm

    def create_empty(self, dataset_name, index, columns, items=None,
                     dtype=np.float32, cls=VariableManager):
        if items is None:
            shape = (len(index), len(columns))
        else:
            shape = (len(index), len(columns), len(items))
        prefix = os.path.join(self.memmap_dir, dataset_name)

        if self.memmap:
            data_path = prefix + '.npy'
            index_path = prefix + '_index.pickle'
            columns_path = prefix + '_columns.pickle'
            items_path = prefix + '_items.pickle'
    
            joblib.dump(list(index), index_path)
            joblib.dump(list(columns), columns_path)
            if items is not None:
                joblib.dump(list(items), items_path)
            data = np.lib.format.open_memmap(
                data_path, mode='w+', dtype=dtype, shape=shape
            )
        else:
            data = np.empty(shape=shape, dtype=dtype)
        dataset = cls(data, index=index, columns=columns, items=items, dtype=dtype)
        dataset._memmap_prefix = prefix
        self.memmap_prefixes[dataset_name] = prefix
        self.memmap_classes[dataset_name] = cls
        return dataset

    def remove_all(self):
        if self.is_tmp and os.path.isdir(self.memmap_dir):
            shutil.rmtree(self.memmap_dir)
