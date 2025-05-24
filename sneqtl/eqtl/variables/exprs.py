import numpy as np

from .base import VariableManager
from ...io.mtx import read_expression_matrix


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

    @classmethod
    def from_matrix_market(cls, exprs_mat_dir, cb_whitelist,
                           rel_min_cells_exprs=0.02, abs_min_cells_exprs=100,
                           min_variance=1e-4, memmap=False,
                           memmap_path_prefix=None, dtype=np.float32):

        if memmap and memmap_path_prefix is None:
            raise ValueError(
                'when memmap is True, a memmap_path_prefix must be supplied'
            )
        exprs = read_expression_matrix(
            exprs_mat_dir, cb_whitelist,
            rel_min_cells_exprs, abs_min_cells_exprs,
            min_variance, dtype=dtype,
        )
        exprs = cls.from_dataframe(exprs, dtype)
        if not memmap:
            return exprs
        return exprs.memmap(memmap_path_prefix)