import os
from functools import cached_property
import joblib
import numpy as np

from snco.records import PredictionRecords
from .base import VariableManager


class HaplotypeVariables(VariableManager):

    index_name = 'cb'
    columns_name = 'haplotypes'

    def __init__(self, data, index, columns, items=None, name=None, dtype=None, frozen=True):
        data = np.clip(data, a_max=1, a_min=0)
        super().__init__(data, index, columns, items=None, name=None, dtype=None, frozen=True)
        self.bin_size = None
        self.chroms = set(chrom for chrom, pos in self.columns)

    @property
    def barcodes(self):
        return self.index

    @property
    def haplotypes(self):
        return self.columns

    def iter_chroms(self):
        haplotypes = self.columns
        chroms = set(chrom for chrom, pos in haplotypes)
        for chrom in chroms:
            yield chrom, self.reindex(columns=[h for h in haplotypes if h[0] == chrom])

    @classmethod
    def read_prediction_records(cls, json_fn, cb_whitelist=None,
                                barcode_query=None, dtype=np.float32):
        prediction_records = PredictionRecords.read_json(json_fn, subset=cb_whitelist)
        if barcode_query is not None:
            prediction_records = prediction_records.query(barcode_query)
        barcodes = prediction_records.barcodes
        bin_size = prediction_records.bin_size
        haplotypes = [
            (chrom, i * bin_size)
            for chrom in prediction_records.chrom_sizes
            for i in range(prediction_records.nbins[chrom])
        ]

        data = np.empty(shape=(len(barcodes), len(haplotypes)))
        offset = 0
        for chrom, chrom_nbins in prediction_records.nbins.items():
            chrom_data = prediction_records._records[:, chrom].stack_values()
            data[:, offset: offset + chrom_nbins] = chrom_data
            offset += chrom_nbins
        haplotype_variables = cls(data, index=barcodes, columns=haplotypes)
        haplotype_variables.bin_size = bin_size
        return haplotype_variables

    def _get_closest_haplotype(self, chrom, pos):
        chrom_positions = [p for c, p in self.columns if c == chrom]
        pos_idx = np.searchsorted(chrom_positions, pos, side='right') - 1
        return chrom, chrom_positions[pos_idx]

    def get_covariate_haplotypes(self, covariate_haplotypes=None,
                                 remove_correlated=True, max_r2=0.95):
        if covariate_haplotypes is None:
            return None, self

        cntrl_hap_columns = [self._get_closest_haplotype(*h) for h in covariate_haplotypes]
        data = self.loc[:, cntrl_hap_columns]

        control_haplotypes = VariableManager(
            data,
            index=self.index,
            columns=cntrl_hap_columns
        )
        if not remove_correlated:
            return control_haplotypes, self

        filtered = self.filter_correlated(control_haplotypes, max_r2)
        return control_haplotypes, filtered
