import os
import joblib
import numpy as np

from snco.records import PredictionRecords
from .base import VariableManager

class HaplotypeVariables(VariableManager):

    @property
    def barcodes(self):
        return self.index

    @property
    def haplotypes(self):
        return self.columns

    @classmethod
    def from_prediction_records(cls, prediction_records, dtype=np.float32):
        if not isinstance(prediction_records, PredictionRecords):
            raise TypeError('prediction_records must be a snco.records.PredictionRecords object')
        barcodes = prediction_records.barcodes
        bin_size = prediction_records.bin_size
        haplotypes = [
            (chrom, i * bin_size)
            for chrom in prediction_records.chrom_sizes
            for i in range(prediction_records.nbins[chrom])
        ]

        data = np.empty(shape=(len(barcodes), len(haplotypes)))
        offset = 0
        for chrom, chrom_nbins in prediction_records.nbins.items():
            chrom_data = prediction_records._records[:, chrom].stack_values()
            data[:, offset: offset + chrom_nbins] = chrom_data
            offset += chrom_nbins
        return cls(data, index=barcodes, columns=haplotypes)


class InteractingVariables(VariableManager):

    @property
    def barcodes(self):
        return self.index

    @property
    def variable_names(self):
        return self.columns


class HaplotypeInteractionVariables:

    def __init__(self, data, index, haplotypes, variable_names):
        self._data = data
        self._variable_names = variable_names
        N = len(variable_names)
        self._haplo_column_map = {
            hap: slice(i * N, (i + 1) * N) for i, hap in enumerate(haplotypes)
        }
        self._index_map = {k: i for i, k in enumerate(index)}
        self._column_map = {i: i for i in range(len(haplotypes) * N)}
        self.is_memmap = isinstance(data, np.memmap)

    @property
    def barcodes(self):
        return tuple(self._index_map.keys())

    @property
    def haplotypes(self):
        return tuple(self._haplo_column_map.keys())

    @property
    def variable_names(self):
        return tuple(self._variable_names)

    def __getitem__(self, haplotype):
        return self._data[:, self._haplo_column_map[haplotype]]

    def iter_haplotypes(self):
        for hap, col_slice in self._haplo_column_map.items():
            yield hap, self._data[:, col_slice]

    @classmethod
    def _memmap_cross_product(cls, haplotype_variables, interacting_variables,
                              memmap_path_prefix, memmap_load=False):

        n_barcodes, n_haplotypes = haplotype_variables.shape
        interaction_matrix = np.ones((n_barcodes, 1), dtype=np.float32)
        variable_names = ['haplo']
        for iv in interacting_variables:
            interaction_matrix = np.einsum(
                'ij,ik->ijk',
                interaction_matrix,
                iv._data
            ).reshape(n_barcodes, -1)
            variable_names = [f'{v1}_{v2}' for v1 in variable_names for v2 in iv.columns]

        n_interactions = interaction_matrix.shape[1]
        total_cols = n_haplotypes * n_interactions

        data_path = memmap_path_prefix + '.npy'
        index_path = memmap_path_prefix + '_index.pickle'
        columns_path = memmap_path_prefix + '_columns.pickle'

        mmap = np.lib.format.open_memmap(
            data_path,
            dtype=np.float32,
            mode='w+',
            shape=(n_barcodes, total_cols)
        )

        col_offset = 0
        for h in haplotype_variables.haplotypes:
            hap_vec = haplotype_variables[h].reshape(-1, 1)
            block = hap_vec * interaction_matrix
            mmap[:, col_offset:col_offset + n_interactions] = block
            col_offset += n_interactions

        mmap.flush()
        del mmap

        joblib.dump(haplotype_variables.index, index_path)
        joblib.dump((haplotype_variables.columns, variable_names), columns_path)


        if memmap_load:
            return cls.load_memmap(memmap_path_prefix)
        return None

    @classmethod
    def from_cross_product(cls, haplotype_variables, interacting_variables,
                           memmap=False, memmap_path_prefix=None, memmap_load=False):

        if not isinstance(haplotype_variables, HaplotypeVariables):
            raise ValueError('haplotype_variables should be of type HaplotypeVariables')

        for iv in interacting_variables:
            if not isinstance(iv, InteractingVariables):
                raise ValueError('All interacting_variables should be InteractingVariables')
            if haplotype_variables.index != iv.index:
                raise ValueError('Index mismatch between haplotypes and interacting variables')

        if memmap:
            return cls._memmap_cross_product(
                haplotype_variables, interacting_variables, memmap_path_prefix, memmap_load
            )

        interaction_matrix = np.ones((haplotype_variables.shape[0], 1), dtype=np.float32)
        variable_names = ['haplo']
        for iv in interacting_variables:
            interaction_matrix = np.einsum('ij,ik->ijk', interaction_matrix, iv._data)
            interaction_matrix = interaction_matrix.reshape(haplotype_variables.shape[0], -1)
            variable_names = [f'{v1}_{v2}' for v1 in variable_names for v2 in iv.columns]

        blocks = []
        for h in haplotype_variables.haplotypes:
            hap_vec = haplotype_variables[h].reshape(-1, 1)
            blocks.append(hap_vec * interaction_matrix)

        final_matrix = np.concatenate(blocks, axis=1)
        return cls(
            final_matrix,
            index=haplotype_variables.index,
            haplotypes=haplotype_variables.columns,
            variable_names=variable_names
        )

    @classmethod
    def load_memmap(cls, path_prefix: str):
        data = np.load(path_prefix + '.npy', mmap_mode='r')
        index = joblib.load(path_prefix + '_index.pickle')
        haplotypes, variable_names = joblib.load(path_prefix + '_columns.pickle')
        return cls(data, index=index, haplotypes=haplotypes, variable_names=variable_names)
