import numpy as np
from .base import VariableManager


class CisHaplotypeCovariates(VariableManager):

    @property
    def gene_ids(self):
        return self.columns

    @property
    def barcodes(self):
        return self.index

    @classmethod
    def from_gene_locs(cls, gene_locs, haplotype_variables):
        gene_ids = list(gene_locs)
        haplotypes = haplotype_variables.haplotypes
        data_idx = []
        for gid in gene_ids:
            gene_chrom, gene_pos = gene_locs[gid]
            chrom_haps = np.array(
                [pos for chrom, pos in haplotypes if chrom == gene_chrom]
            )
            pos_idx = np.searchsorted(chrom_haps, gene_pos, side='right') - 1
            data_idx.append((gene_chrom, chrom_haps[pos_idx]))
        data = haplotype_variables.loc[:, data_idx]
        return cls(
            data,
            index=haplotype_variables.index,
            columns=gene_ids
        )

    def get_cis_haplotype(self, gene_id):
        col_idx = self._column_map[gene_id]
        return self._data[:, col_idx]