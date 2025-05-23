import os
import joblib
import numpy as np

from snco.records import PredictionRecords
from .base import VariableManager

class HaplotypeVariables(VariableManager):

    @property
    def barcodes(self):
        return self.index

    @property
    def haplotypes(self):
        return self.columns

    @classmethod
    def from_prediction_records(cls, prediction_records, dtype=np.float32):
        if not isinstance(prediction_records, PredictionRecords):
            raise TypeError('prediction_records must be a snco.records.PredictionRecords object')
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
        return cls(data, index=barcodes, columns=haplotypes)
