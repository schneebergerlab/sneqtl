import itertools as it
from functools import cached_property
import numpy as np
from .base import VariableManager
from .arr_utils import get_dummies


class ParentalGenotypesContainer:

    def __init__(self, hap1, hap2, reference_genotype=None):
        if hap1.shape != hap2.shape or hap1.index != hap2.index or hap1.columns != hap2.columns:
            raise ValueError('haplotype variables must have identical shapes, indexes and columns')

        self._hap1 = hap1
        self._hap2 = hap2
        if reference_genotype is None:
            self._ref_geno = self._hap1.columns[0]
        else:
            self._ref_geno = reference_genotype

    @property
    def index(self):
        return self._hap1.index

    @property
    def columns(self):
        return self._hap1.columns

    def as_covariates(self):
        haploid_genotypes = VariableManager(
            self._hap1._data + self._hap2._data,
            index=self._hap1.index,
            columns=self._hap1.columns,
        ).drop_uninformative(axis=1)
        haploid_genotypes = haploid_genotypes.drop(columns=[self._ref_geno])
        if haploid_genotypes.has_row_sum_constraint:
            haploid_genotypes = haploid_genotypes.reindex(columns=haploid_genotypes.columns[1:])
        return haploid_genotypes

    def get_haplotype_variables(self, haplotype):
        haplotype = haplotype.reshape(-1, 1)
        hap1_vars = self._hap1._data * (1 - haplotype)
        hap2_vars = self._hap2._data * haplotype
        hap_var_data = hap1_vars + hap2_vars
        columns=[f'haplo_{geno}' for geno in self._hap1.columns]

        hap_vars = VariableManager(
            hap_var_data,
            index=self._hap1.index,
            columns=columns
        ).drop_uninformative(axis=1)
        hap_vars = hap_vars.drop(columns=[f'haplo_{self._ref_geno}'])
        if hap_vars.has_row_sum_constraint:
            hap_vars = hap_vars.reindex(columns=hap_vars.columns[1:])
        return hap_vars

    @classmethod
    def from_genotype_dict(cls, parental_genotypes: dict, reference_genotype=None):
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

        return cls(hap1=hap1, hap2=hap2, reference_genotype=reference_genotype)

    def reindex(self, index=None, columns=None):
        hap1 = self.hap1.reindex(index, columns)
        hap2 = self.hap2.reindex(index, columns)
        return self.__class__(hap1, hap2)
