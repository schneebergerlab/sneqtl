import numpy as np
import joblib

from .base import VariableManager
from .genotypes import ParentalGenotypesContainer
from .haplotypes import HaplotypeVariables
from .variants import VariantVariables


class HaplotypeInteractionVariables(VariableManager):

    @property
    def barcodes(self):
        return self.columns

    @property
    def haplotypes(self):
        return self.index

    @property
    def variable_names(self):
        return self.items

    def iter_haplotypes(self):
        for hap, i in self._index_map.items():
            yield hap, self._data[i, :, ...]

    @classmethod
    def from_cross_product(cls, haplotype_variables,
                           parental_genotypes, variant_variables=None, interacting_variables=None,
                           name=None, memmap_manager=None, dtype=np.float32):

        if haplotype_variables.index != parental_genotypes.index:
            raise ValueError('Index mismatch between haplotypes and parental genotypes')

        if variant_variables is not None:
            if variant_variables.columns != parental_genotypes.columns:
                raise ValueError('Column mismatch between variants and parental genotypes')

        if interacting_variables is None:
            interacting_variables = []
        for iv in interacting_variables:
            if not isinstance(iv, VariableManager):
                raise ValueError(
                    'All interacting_variables should be of type VariableManager'
                )
            if haplotype_variables.index != iv.index:
                raise ValueError('Index mismatch between haplotypes and interacting variables')

        n_barcodes, n_haplo_positions = haplotype_variables.shape[:2]
        interaction_matrix = np.ones((n_barcodes, 1), dtype=np.float32)
        if variant_variables is None:
            base_variable_names = [
                f'haplo_{g}'
                for g in parental_genotypes.columns
                if g != parental_genotypes._ref_geno
            ]
        else:
            base_variable_names = ['alt_snp']

        variable_names = base_variable_names

        for iv in interacting_variables:

            interaction_matrix = np.einsum(
                'ij,ik->ijk', interaction_matrix, iv._data
            ).reshape(n_barcodes, -1)

            variable_names = [
                f'{v1}_{v2}'
                for v1 in variable_names
                for v2 in iv.columns
            ]

        n_interactions = len(variable_names)
        n_test_positions = n_haplo_positions if variant_variables is None else variant_variables.shape[0]
        ix_columns = haplotype_variables.columns if variant_variables is None else variant_variables.index

        if memmap_manager is not None:
            cross_prod_results = memmap_manager.create_empty(
                name,
                index=ix_columns,
                columns=haplotype_variables.index,
                items=variable_names,
                dtype=dtype,
                cls=cls
            )
        else:
            cross_prod_results = cls(
                np.empty(shape=(n_test_positions, n_barcodes, n_interactions), dtype=dtype),
                index=ix_columns,
                columns=haplotype_variables.index,
                items=variable_names,
                name=name,
                dtype=dtype,
            )
        cp_arr = cross_prod_results._data

        for (chrom, pos), i in haplotype_variables._column_map.items():
            hap = haplotype_variables[(chrom, pos)]

            if variant_variables is not None:
                var = variant_variables.get_bin_variants(chrom, pos)
            else:
                var = None

            # shape (n_snps, n_barcodes, 1) if var is not None, else shape (n_barcodes, n_features)
            hap_vars = parental_genotypes.get_haplotype_variables(hap, variants=var)

            if variant_variables is None:
                hv = np.einsum(
                    'ij,ik->ijk', hap_vars._data, interaction_matrix
                ).reshape(n_barcodes, n_interactions)

                cp_arr[i, :, :] = hv

            else:
                idx = [variant_variables._index_map[v] for v in hap_vars.index]

                hv = np.einsum(
                    'xij,ik->xijk', hap_vars._data, interaction_matrix
                ).reshape(len(idx), n_barcodes, n_interactions)

                cp_arr[idx, :, :] = hv

        return cross_prod_results