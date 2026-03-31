import numpy as np
import joblib

from .base import VariableManager
from .genotypes import ParentalGenotypesContainer
from .haplotypes import HaplotypeVariables
from .variants import VariantVariables


class HaplotypeInteractionVariables(VariableManager):

    @property
    def barcodes(self):
        return self.index

    @property
    def haplotypes(self):
        return self.columns

    @property
    def variable_names(self):
        return self.items

    def iter_haplotypes(self):
        for hap, i in self._column_map.items():
            yield hap, self._data[:, i, ...]

    @classmethod
    def from_cross_product(cls, haplotype_variables,
                           parental_genotypes=None, variant_variables=None, interacting_variables=None,
                           name=None, memmap_manager=None, dtype=np.float32):

        #if not isinstance(haplotype_variables, HaplotypeVariables):
        #    raise ValueError('haplotype_variables should be of type HaplotypeVariables')

        if parental_genotypes is not None: 
        #    if not isinstance(parental_genotypes, ParentalGenotypesContainer):
        #        raise ValueError('parental_genotypes should be of type ParentalGenotypesContainer')
            if haplotype_variables.index != parental_genotypes.index:
                raise ValueError('Index mismatch between haplotypes and parental genotypes')
        if variant_variables is not None:
            if parental_genotypes is None:
                raise ValueError('variant_variables can only be supplied in conjunction with parental_genotypes')
        #    if not isinstance(variant_variables, VariantVariables):
        #        raise ValueError('variant_variables should be of type VariantVariables')
            if variant_variables.columns != parental_genotypes.columns:
                raise ValueError('Column mismatch between haplotypes and parental genotypes')
            

        if interacting_variables is None:
            interacting_variables = []
        for iv in interacting_variables:
            if not isinstance(iv, VariableManager):
                raise ValueError(
                    'All interacting_variables should be of type VariableManager'
                )
            if haplotype_variables.index != iv.index:
                raise ValueError('Index mismatch between haplotypes and interacting variables')

        # use the first bin to make variables and determine which are present
        if parental_genotypes is not None:
            test_hap = haplotype_variables.columns[0]
            if variant_variables is None:
                hv = parental_genotypes.get_haplotype_variables(
                    haplotype_variables.loc[:, test_hap], variants=None
                )
                if hv.is_singular:
                    raise ValueError(
                        'The haplotype variable matrix is singular, consider altering experimental design'
                    )
                variable_names = hv.columns
                hv = hv._data
            else:
                hv = parental_genotypes.get_haplotype_variables(
                    haplotype_variables.loc[:, test_hap],
                    variants=variant_variables.get_bin_variants(*test_hap)
                )
                variable_names = hv.items
                hv = hv._data[0, ...]
        else:
            hv = haplotype_variables.iloc[:, 0].reshape(-1, 1)
            variable_names = ['haplo']

        # precompute the interaction matrix from all the variables
        # except parental genotype (which is a bit more complex)
        n_barcodes, n_haplo_positions = haplotype_variables.shape
        interaction_matrix = np.ones((haplotype_variables.shape[0], 1), dtype=np.float32)
        for iv in interacting_variables:
            interaction_matrix = np.einsum('ij,ik->ijk', interaction_matrix, iv._data).reshape(n_barcodes, -1)
            variable_names = [f'{v1}_{v2}' for v1 in variable_names for v2 in iv.columns]

        # test singularity
        hv = np.einsum('ij,ik->ijk', hv, interaction_matrix).reshape(n_barcodes, -1)
        if np.linalg.matrix_rank(hv, tol=1e-10) < hv.shape[1]:
            raise ValueError(
                'The haplotype variable matrix is singular, consider altering experimental design'
            )

        n_interactions = len(variable_names)
        n_test_positions = n_haplo_positions if variant_variables is None else variant_variables.shape[0]
        ix_columns = haplotype_variables.columns if variant_variables is None else variant_variables.index

        if memmap_manager is not None:
            cross_prod_results = memmap_manager.create_empty(
                name,
                index=haplotype_variables.index,
                columns=ix_columns,
                items=variable_names,
                dtype=dtype,
                cls=cls
            )
        else:
            cross_prod_results = cls(
                np.empty(shape=(n_barcodes, n_test_positions, n_interactions), dtype=dtype),
                index=haplotype_variables.index,
                columns=ix_columns,
                items=variable_names,
                name=name,
                dtype=dtype,
            )
        cp_arr = cross_prod_results._data

        for (chrom, pos), i in haplotype_variables._column_map.items():
            hap = haplotype_variables[(chrom, pos)]
            if parental_genotypes is not None:
                if variant_variables is not None:
                    var = variant_variables.get_bin_variants(chrom, pos)
                else:
                    var = None
                # shape (n_snps, n_barcodes, 1) if var is not None, else shape (n_barcodes, n_features)
                hap_vars = parental_genotypes.get_haplotype_variables(hap, variants=var)
            else:
                # no genotype features, shape (n_barcodes, 1)
                hap_vars = hap.reshape(n_barcodes, 1)

            if hap_vars.ndim == 2:
                idx = i
                if not isinstance(hap_vars, np.ndarray):
                    hap_vars = hap_vars._data
                hv = np.einsum('ij,ik->ijk', hap_vars, interaction_matrix)
                hv = hv.reshape(n_barcodes, n_interactions)
            else:
                idx = [variant_variables._index_map[v] for v in hap_vars.index]
                hv = np.einsum('xij,ik->ixjk', hap_vars._data, interaction_matrix)
                hv = hv.reshape(n_barcodes, var.shape[0], n_interactions)
            cp_arr[:, idx, ...] = hv

        return cross_prod_results
