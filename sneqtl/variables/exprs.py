import os
import numpy as np
from scipy.io import mmread, mmwrite

from .base import VariableManager, SparseMixIn


class ExpressionVariables(VariableManager, SparseMixIn):

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
    def read_matrix_market(cls, exprs_mat_dir, cb_whitelist=None, name=None,
                           dtype=np.float32, sparse=False):
        data = super().read_matrix_market(
            mtx_dir=exprs_mat_dir,
            name=name,
            dtype=dtype,
            mtx_fn='matrix.mtx',
            index_fn='features.tsv',
            columns_fn='barcodes.tsv',
        )
        if cb_whitelist is not None:
            data = data.reindex(columns=cb_whitelist)
        if not sparse:
            data = data.as_dense()
        return data

    def filter_genes(self,
                     rel_min_cells_exprs=0.02,
                     abs_min_cells_exprs=100,
                     min_variance=1e-4):
        data = self._data
        min_cells_exprs = max(data.shape[1] * rel_min_cells_exprs, abs_min_cells_exprs)
        mask = np.logical_and(
            (data > 0).sum(axis=1) > min_cells_exprs,
            data.var(axis=1) >= min_variance
        )

        filtered_index = [k for k, keep in zip(self._index_map, mask) if keep]
        return self.reindex(index=filtered_index)

    def log_normalise(self, alpha=0.25):
        barcode_counts = self._data.sum(axis=0)
        sf = barcode_counts / barcode_counts.mean()
        ln_data = np.log2(self._data / sf + (1.0 / 4 * alpha))
        ln_data[~np.isfinite(ln_data)] = 0
        return ExpressionVariables(
            data=ln_data,
            index=self.index,
            columns=self.barcodes,
            dtype=ln_data.dtype,
            name=self.name
        )
