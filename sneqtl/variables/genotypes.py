import numpy as np
from .base import VariableManager
from .arr_utils import get_dummies

class ParentalGenotypesContainer:

    def __init__(self, hap1, hap2):
        if hap1.shape != hap2.shape or hap1.index != hap2.index or hap1.columns != hap2.columns:
            raise ValueError('haplotype variables must have identical shapes, indexes and columns')

        self.genotype_names = hap1.columns
        self.genotypes = VariableManager(
            hap1._data + hap2._data,
            index=hap1.index,
            columns=hap1.columns
        ).drop_uninformative()
        self.hap1 = hap1.reindex(columns=self.genotypes.columns)
        self.hap2 = hap2.reindex(columns=self.genotypes.columns)

    @property
    def index(self):
        return self.genotypes.index

    @property
    def columns(self):
        return self.genotypes.columns

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

    def as_covariates(self):
        return self.genotypes.reindex(columns=list(self.columns[1:]))
