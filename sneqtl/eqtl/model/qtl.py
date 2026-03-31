from copy import copy

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from .lm import fit_models
from sneqtl.variables.covariates import Covariates

DEFAULT_RANDOM_SEED = 101
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


class SingleGeneResults:

    _base_schema = [
        pa.field('gene_id', pa.string()),
        pa.field('chrom', pa.string()),
        pa.field('pos', pa.int32()),
        pa.field('lod_score', pa.float64()),
        pa.field('lrt_pval', pa.float64()),
    ]

    def __init__(self, gene_id, bin_size, tested_haplotypes, variable_names,
                 lrt_pval_threshold, param_pval_thresholds, data=None):
        self.gene_id = gene_id
        self.ntests = len(tested_haplotypes)
        self.bin_size = bin_size
        self.lrt_pval_threshold = lrt_pval_threshold
        self.variable_names = variable_names
        self.param_pval_thresholds = dict(zip(variable_names, param_pval_thresholds))

        fields = copy(self._base_schema)
        # add bin size to pos field
        fields[2] = fields[2].with_metadata({'bin_size': str(bin_size)})
        # add minP_threshold to lrt_pval field
        fields[4] = fields[4].with_metadata({'threshold': str(float(lrt_pval_threshold))})

        if len(tested_haplotypes[0]) == 3:
            fields.insert(3, pa.field('snp_id', pa.string()))
            chrom, pos, snp = zip(*tested_haplotypes)
            self._chrom_array = pa.array(chrom, pa.string())
            self._pos_array = pa.array(pos, pa.int32())
            self._si_array = pa.array(snp, pa.string())
        else:
            chrom, pos = zip(*tested_haplotypes)
            self._chrom_array = pa.array(chrom, pa.string())
            self._pos_array = pa.array(pos, pa.int32())
            self._si_array = None

        # param pval fields have thresholds added as metadata
        pval_fields = []
        for vn, p_t in self.param_pval_thresholds.items():
            pval_fields.append(
                pa.field(
                    f'{vn}_pval',
                    pa.float64(),
                    metadata={'threshold': str(float(p_t))}
                )
            )

        self.schema = pa.schema(
            fields +
            [pa.field(f'{vn}_lod_score', pa.float64()) for vn in variable_names] +
            [pa.field(f'{vn}_coef', pa.float64()) for vn in variable_names] +
            pval_fields,
        )
        if data is None:
            self.table = None
        else:
            self.populate(data)

    def populate(self, data):
        arrays = [
            pa.array([self.gene_id] * self.ntests, type=pa.string()),
            self._chrom_array,
            self._pos_array,
        ]

        if self._si_array is not None:
            arrays.append(self._si_array)

        for arr in data.T:
            arrays.append(pa.array(arr, type=pa.float64()))

        self.table = pa.Table.from_arrays(arrays, schema=self.schema)

    def write_parquet(self, fn):
        if self.table is None:
            raise ValueError("Data has not yet been populated")

        pq.write_table(
            self.table,
            fn,
            compression="zstd"
        )

    @classmethod
    def from_table(cls, table):
        gene_ids = table.column("gene_id").unique().to_pylist()
        if len(gene_ids) != 1:
            raise ValueError(f"Inconsistent gene_id column in {fn}: {gene_ids}")
        gene_id = gene_ids[0]

        obj = cls.__new__(cls)  # bypass __init__
        obj.gene_id = gene_id
        obj.ntests = table.num_rows
        obj.schema = table.schema
        obj.table = table
        
        obj.bin_size = int(table.schema.field('pos').metadata.get(b'bin_size'))
        obj.lrt_pval_threshold = float(table.schema.field('lrt_pval').metadata.get(b'threshold'))
        obj.param_pval_thresholds = {}
        for fname in table.schema.names:
            if fname.endswith('_pval') and fname != 'lrt_pval':
                obj.param_pval_thresholds[fname] = float(table.schema.field(fname).metadata.get(b'threshold'))

        # keep these for compatibility with existing code
        obj._chrom_array = table.column("chrom")
        obj._pos_array = table.column("pos")
        obj._si_array = table.column("snp_id") if "snp_id" in obj.schema.names else None

        return obj

    @classmethod
    def read_parquet(cls, fn):
        table = pq.read_table(fn)
        return cls.from_table(table)

    def to_dataframe(self):
        df = self.table.to_pandas()
        df.attrs = {
            'lrt_pval_threshold': self.lrt_pval_threshold,
            'param_pval_thresholds': self.param_pval_thresholds,
        }
        return df


class SinglePhenoQTLAnalysis:

    def __init__(self, y, haplotype_variables, gene_id, covariates=None,
                 minP_threshold=0.01, n_perm=1000, bonf_correction=1.0, bin_size=25_000,
                 L_precomp=None, diag_precomp=None):
        self.y = y
        self.gene_id = gene_id
        self.haplotype_variables = haplotype_variables
        if covariates is None:
            self.covariates = Covariates.null_covariates(haplotype_variables)
        else:
            self.covariates = covariates

        # precomputed matrices to save compute across many phenotypes
        self.L_precomp = L_precomp
        self.diag_precomp = diag_precomp

        # multiple testing correction params
        self.minP_threshold = minP_threshold
        self.n_perm = n_perm
        self.bonf_correction = bonf_correction
        self.bin_size = bin_size
        self.results = None

    def test_all_positions(self, batch_size=500, rng=DEFAULT_RNG):
        results, lrt_pval_threshold, param_pval_thresholds, *_ = fit_models(
            self.y,
            self.covariates._data,
            self.haplotype_variables._data,
            batch_size=batch_size,
            n_perm=self.n_perm,
            minP_threshold=self.minP_threshold,
            bonf_correction=self.bonf_correction,
            L_precomp=self.L_precomp._data,
            diag_precomp=self.diag_precomp._data,
            rng=rng
        )
        self.results = SingleGeneResults(
            self.gene_id,
            bin_size=self.bin_size,
            tested_haplotypes=self.haplotype_variables.haplotypes,
            variable_names=self.haplotype_variables.variable_names,
            lrt_pval_threshold=lrt_pval_threshold,
            param_pval_thresholds=param_pval_thresholds,
        )
        self.results.populate(results)
        return self.results
