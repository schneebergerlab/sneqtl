import numpy as np

from .base import VariableManager


class SingleGeneResults(VariableManager):

    @property
    def haplotypes(self):
        return self.index

    @property
    def results_colnames(self):
        return self.columns


class eQTLResults(VariableManager):

    @property
    def gene_ids(self):
        return self.index

    @property
    def haplotypes(self):
        return self.columns

    @property
    def results_colnames(self):
        return self.items
