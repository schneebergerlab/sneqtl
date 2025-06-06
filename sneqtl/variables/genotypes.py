import itertools as it
from functools import cached_property
import numpy as np
from .base import VariableManager
from .arr_utils import get_dummies

class ParentalGenotypesContainer:

    def __init__(self, hap1, hap2):
        if hap1.shape != hap2.shape or hap1.index != hap2.index or hap1.columns != hap2.columns:
            raise ValueError('haplotype variables must have identical shapes, indexes and columns')

        self._hap1 = hap1
        self._hap2 = hap2

    @property
    def index(self):
        return self._hap1.index

    @property
    def columns(self):
        return self._hap1.columns

    @cached_property
    def diploid_covariates(self):

        def _unique_offdiag_outer_flat(arr1, arr2):
            n, d = arr1.shape
            iu_i, iu_j = np.triu_indices(d, k=1)
            il_i, il_j = np.tril_indices(d, k=-1)
            op = arr1[:, :, None] * arr2[:, None, :]
            return op[:, iu_i, iu_j] + op[:, il_i, il_j]

        diploid_geno_data = _unique_offdiag_outer_flat(self._hap1._data, self._hap2._data)
        diploid_columns = [f'{hap1}:{hap2}' for hap1, hap2 in it.combinations(self._hap1.columns, 2)]
        diploid_genotypes = VariableManager(
            data=diploid_geno_data,
            index=self._hap1.index,
            columns=diploid_columns
        ).drop_uninformative(axis=1)
        if diploid_genotypes.has_row_sum_constraint:
            diploid_genotypes = diploid_genotypes.reindex(columns=diploid_genotypes.columns[1:])
        return diploid_genotypes

    @cached_property
    def haploid_covariates(self):
        haploid_genotypes = VariableManager(
            self._hap1._data + self._hap2._data,
            index=self._hap1.index,
            columns=self._hap1.columns,
        ).drop_uninformative(axis=1)
        if haploid_genotypes.has_row_sum_constraint:
            haploid_genotypes = haploid_genotypes.reindex(columns=haploid_genotypes.columns[1:])
        return haploid_genotypes

    def get_haplotype_variables(self, haplotype):
        haplotype = haplotype.reshape(-1, 1)
        phap1, phap2 = self._hap1._data, self._hap2._data
        hap1_vars = np.einsum('ij,ik->ijk', haplotype, phap1)
        hap2_vars = np.einsum('ij,ik->ijk', 1 - haplotype, phap2)
        hap_var_data = (hap1_vars + hap2_vars).reshape(phap1.shape)
        hap_vars = VariableManager(
            hap_var_data,
            index=self._hap1.index,
            columns=[f'haplo_{geno}' for geno in self._hap1.columns]
        )
        if hap_vars.has_row_sum_constraint:
            hap_vars = hap_vars.reindex(columns=hap_vars.columns[1:])
        return hap_vars

    @classmethod
    def from_genotype_dict(cls, parental_genotypes: dict):
        barcodes = list(parental_genotypes.keys())
        genotypes = list(parental_genotypes.values())

        # Split genotypes into hap1 and hap2
        hap1, hap2 = zip(*(g.split(':') for g in genotypes))
        hap1 = np.array(hap1)
        hap2 = np.array(hap2)
        all_haplotypes = np.unique([hap1, hap2])

        # Dummy encode hap1 and hap2
        hap1, _ = get_dummies(hap1, categories=all_haplotypes, drop_first=False)
        hap2, _ = get_dummies(hap2, categories=all_haplotypes, drop_first=False)

        # Check for invalid homozygous genotypes (e.g. col0:col0)
        if np.any((hap1 + hap2) > 1):
            raise ValueError('Some parental genotypes are not hybrids e.g. col0:col0')

        hap1 = VariableManager(hap1, index=barcodes, columns=all_haplotypes)
        hap2 = VariableManager(hap2, index=barcodes, columns=all_haplotypes)

        return cls(hap1=hap1, hap2=hap2)

    def reindex(self, index=None, columns=None):
        hap1 = self.hap1.reindex(index, columns)
        hap2 = self.hap2.reindex(index, columns)
        return self.__class__(hap1, hap2)
