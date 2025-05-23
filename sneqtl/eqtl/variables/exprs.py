import numpy as np

from .base import VariableManager


class ExpressionVariables(VariableManager):

    @property
    def gene_ids(self):
        return self.index

    @property
    def barcodes(self):
        return self.columns

    def iter_genes(self):
        for gene_id, row_idx in self._index_map.items():
            yield gene_id, self._data[row_idx]
