import os
import numpy as np
from scipy.io import mmread, mmwrite

from .base import VariableManager


def read_index(idx_fn):
    with open(idx_fn) as f:
        idx = [r.strip().split('\t')[0] for r in f.readlines()]
    return np.array(idx)


def read_matrix_market(mtx_dir, dtype=np.float32,
                       mtx_fn='matrix.mtx',
                       barcode_fn='barcodes.tsv',
                       feature_fn='features.tsv'):
    features = read_index(os.path.join(mtx_dir, feature_fn))
    barcodes = read_index(os.path.join(mtx_dir, barcode_fn))
    mtx = mmread(os.path.join(mtx_dir, mtx_fn)).astype(dtype, copy=False).tocsr()
    return features, barcodes, mtx


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
    def read_matrix_market(cls, exprs_mat_dir, cb_whitelist=None, name=None,
                           memmap_manager=None, dtype=np.float32):
        gene_ids, barcodes, mtx = read_matrix_market(exprs_mat_dir, dtype=dtype)
        if cb_whitelist is not None:
            barcode_idx = {cb: i for i, cb in enumerate(barcodes)}
            try:
                barcodes, barcode_filt_idx = zip(*[
                    (cb, barcode_idx[cb]) for cb in cb_whitelist if cb in barcode_idx
                ])
            except ValueError:
                raise ValueError('no overlap between barcodes and cb_whitelist')
            mtx = mtx[:, barcode_filt_idx]

        if memmap_manager is not None:
            exprs = memmap_manager.create_empty(
                name,
                index=gene_ids,
                columns=barcodes,
                dtype=dtype,
                cls=cls
            )
            exprs._data[:] = mtx.toarray()
            exprs._data.flush()
        else:
            exprs = cls(
                data=mtx.toarray(),
                index=gene_ids,
                columns=barcodes,
                dtype=dtype,
                name=name
            )
        return exprs

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
