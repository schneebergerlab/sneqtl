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
        return self.loc[:, 'lrt_pval']

    def apply_lod_thresholds(self, lrt_lod_thresh, param_lod_thresh):
        dtype = self._data.dtype

        # Flag for overall LOD significance
        sig_model = (self.loc[:, ['lod_score']] > lrt_lod_thresh).astype(dtype)

        # Identify param LOD columns and construct corresponding sig col names
        param_lod_cols = [c for c in self.columns if c.endswith('_lod_score')]
        param_sig_colnames = [c.rsplit('_', 2)[0] + '_sig' for c in param_lod_cols]

        # Flag for significant parameters
        sig_params = (self.loc[:, param_lod_cols] > param_lod_thresh).astype(dtype)

        # Assemble new data matrix
        start = 2 + len(param_lod_cols)
        data = np.column_stack([
            self._data[:, :2],           # lod_score, lrt_pval
            sig_model,                   # sig_model
            self._data[:, 2:start],      # param LODs
            sig_params,                  # binary flags for param sigs
            self._data[:, start:]        # remaining columns (e.g. coefs, pvals)
        ])

        columns = [
            'lod_score', 'lrt_pval', 'lrt_sig',
            *param_lod_cols, *param_sig_colnames,
            *self.columns[start:]
        ]

        return self.__class__(
            data,
            columns=columns,
            index=self.index,
            name=self.name,
            dtype=dtype
        )

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

