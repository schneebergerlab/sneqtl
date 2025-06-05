import numpy as np
import joblib

from .base import VariableManager
from .genotypes import ParentalGenotypesContainer
from .haplotypes import HaplotypeVariables


class HaplotypeInteractionVariables(VariableManager):

    @property
    def barcodes(self):
        return tuple(self._index_map.keys())

    @property
    def haplotypes(self):
        return tuple(self._column_map.keys())

    @property
    def variable_names(self):
        return tuple(self._item_map.keys())

    def iter_haplotypes(self):
        for hap, i in self._column_map.items():
            yield hap, self._data[:, i, ...]

    @classmethod
    def from_cross_product(cls, haplotype_variables,
                           parental_genotypes=None, interacting_variables=None,
                           name=None, memmap_manager=None, dtype=np.float32):

        if not isinstance(haplotype_variables, HaplotypeVariables):
            raise ValueError('haplotype_variables should be of type HaplotypeVariables')

        if parental_genotypes is not None: 
            if not isinstance(parental_genotypes, ParentalGenotypesContainer):
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

        n_interactions = len(variable_names)

        if memmap_manager is not None:
            cross_prod_results = memmap_manager.create_empty(
                name,
                index=haplotype_variables.index,
                columns=haplotype_variables.columns,
                items=variable_names,
                dtype=dtype,
                cls=cls
            )
        else:
            cross_prod_results = cls(
                np.empty(shape=(n_barcodes, n_haplo_positions, n_interactions), dtype=dtype),
                index=haplotype_variables.index,
                columns=haplotype_variables.columns,
                items=variable_names,
                name=name,
                dtype=dtype,
            )
        cp_arr = cross_prod_results._data

        for h, i in haplotype_variables._column_map.items():
            hap = haplotype_variables[h].reshape(-1, 1)
            if parental_genotypes is not None:
                hap1_vars = np.einsum('ij,ik->ijk', hap, parental_genotypes.hap1._data)
                hap2_vars = np.einsum('ij,ik->ijk', 1 - hap, parental_genotypes.hap2._data)
                hap_vars = (hap1_vars + hap2_vars).reshape(n_barcodes, n_iteractions)
            else:
                hap_vars = hap
            hap_vars = np.einsum('ij,ik->ijk', hap_vars, interaction_matrix)
            hap_vars = hap_vars.reshape(n_barcodes, n_interactions)
            cp_arr[:, i, ...] = hap_vars

        return cross_prod_results
