import numpy as np
import pandas as pd


def read_cb_stats(cb_stats_fn, cb_filter_exprs=None):
    cb_stats = pd.read_csv(
        cb_stats_fn, index_col='cb', sep='\t',
    )
    if cb_filter_exprs is not None:
        cb_stats = cb_stats.query(cb_filter_exprs)
    return cb_stats


def read_eqtl_results(eqtl_res_fn):
    eqtl_results = pd.read_csv(
        eqtl_res_fn,
        sep='\t',
        dtype={
            'chrom': str,
            'pos': int,
            'lod_score': float,
            'pval': float,
        }
    )
    test_gene_id, test_chrom = eqtl_results.iloc[0].loc[['gene_id', 'chrom']]
    test_pos = eqtl_results.query('gene_id == @test_gene_id & chrom == @test_chrom').pos.values
    bin_size = int(np.unique(np.diff(np.sort(test_pos))))
    return eqtl_results, bin_size