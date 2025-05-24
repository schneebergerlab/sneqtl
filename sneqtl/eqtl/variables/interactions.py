import numpy as np
import joblib

from .base import VariableManager
from .genotypes import ParentalGenotypesContainer
from .haplotypes import HaplotypeVariables


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
    def from_cross_product(cls, haplotype_variables,
                           parental_genotypes=None, interacting_variables=None,
                           memmap=False, memmap_path_prefix=None):

        if not isinstance(haplotype_variables, HaplotypeVariables):
            raise ValueError('haplotype_variables should be of type HaplotypeVariables')

        if parental_genotypes is not None and not isinstance(parental_genotypes, ParentalGenotypesContainer):
            raise ValueError('parental_genotypes should be of type ParentalGenotypesContainer')
        if haplotype_variables.index != parental_genotypes.genotypes.index:
            raise ValueError('Index mismatch between haplotypes and parental genotypes')

        if interacting_variables is None:
            interacting_variables = []
        for iv in interacting_variables:
            if not isinstance(iv, VariableManager):
                raise ValueError(
                    'All interacting_variables should be of type VariableManager'
                )
            if haplotype_variables.index != iv.index:
                raise ValueError('Index mismatch between haplotypes and interacting variables')

        if memmap and memmap_path_prefix is None:
            raise ValueError('When memmap is True, a memmap_path_prefix should be provided')

        if parental_genotypes is not None:
            variable_names = [f'haplo_{geno}' for geno in parental_genotypes.genotype_names]
        else:
            variable_names = ['haplo']

        # precompute the interaction matrix from all the variables
        # except parental genotype (which is a bit more complex)
        n_barcodes, n_haplo_positions = haplotype_variables.shape
        interaction_matrix = np.ones((haplotype_variables.shape[0], 1), dtype=np.float32)
        for iv in interacting_variables:
            interaction_matrix = np.einsum('ij,ik->ijk', interaction_matrix, iv._data)
            interaction_matrix = interaction_matrix.reshape(n_barcodes, -1)
            variable_names = [f'{v1}_{v2}' for v1 in variable_names for v2 in iv.columns]

        # calculate the total size of cross product array we need to product
        n_interactions = len(variable_names)
        total_cols = n_interactions * n_haplo_positions

        if memmap:
            data_path = memmap_path_prefix + '.npy'
            index_path = memmap_path_prefix + '_index.pickle'
            columns_path = memmap_path_prefix + '_columns.pickle'

            joblib.dump(haplotype_variables.index, index_path)
            joblib.dump((haplotype_variables.columns, variable_names), columns_path)

            # create a memory mapped results matrix to insert the results into
            cross_prod_results = np.lib.format.open_memmap(
                data_path,
                dtype=np.float32,
                mode='w+',
                shape=(n_barcodes, total_cols)
            )
        else:
            # otherwise we create a large empty array to insert the results into
            cross_prod_results = np.empty(
                shape=(n_barcodes, total_cols),
                dtype=np.float32
            )

        col_offset = 0
        for h in haplotype_variables.haplotypes:
            hap = haplotype_variables[h].reshape(-1, 1)
            if parental_genotypes is not None:
                hap1_vars = np.einsum('ij,ik->ijk', hap, parental_genotypes.hap1._data)
                hap2_vars = np.einsum('ij,ik->ijk', 1 - hap, parental_genotypes.hap2._data)
                hap_vars = (hap1_vars + hap2_vars).reshape(n_barcodes, -1)
            else:
                hap_vars = hap
            hap_vars = np.einsum('ij,ik->ijk', hap_vars, interaction_matrix).reshape(n_barcodes, -1)
            cross_prod_results[:, col_offset:col_offset + n_interactions] = hap_vars
            col_offset += n_interactions

        if not memmap:
            return cls(
                cross_prod_results,
                index=haplotype_variables.index,
                haplotypes=haplotype_variables.columns,
                variable_names=variable_names
            )

        # for memmapped arrays, we have some cleaning up to do
        cross_prod_results.flush()
        del cross_prod_results
        # reload as read only
        return cls.load_memmap(memmap_path_prefix)

    @classmethod
    def load_memmap(cls, path_prefix: str):
        data = np.load(path_prefix + '.npy', mmap_mode='r')
        index = joblib.load(path_prefix + '_index.pickle')
        haplotypes, variable_names = joblib.load(path_prefix + '_columns.pickle')
        return cls(data, index=index, haplotypes=haplotypes, variable_names=variable_names)
