import itertools as it

import numpy as np
import pandas as pd

from .base import VariableManager


class SingleGeneResults(VariableManager):

    @property
    def haplotypes(self):
        return self.index

    @property
    def results_colnames(self):
        return self.columns

    @property
    def lod_score(self):
        return self.loc[:, 'lod_score']

    @property
    def pval(self):
        return self.loc[:, 'pval']

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

    def to_dataframe(self):
        data = self._data.reshape(-1, self.shape[2])
        index = pd.MultiIndex.from_tuples(
            [(gid, chrom, pos) for gid, (chrom, pos)
             in it.product(self.gene_ids, self.haplotypes)],
            names=['gene_id', 'chrom', 'pos']
        )
        res = pd.DataFrame(data, index=index, columns=self.results_colnames)
        return res.reset_index()

