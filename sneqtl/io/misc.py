import numpy as np
import pandas as pd


def read_cb_stats(cb_stats_fn, cb_filter_exprs=None):
    cb_stats = pd.read_csv(
        cb_stats_fn, index_col='cb', sep='\t',
    )
    # hack, needs to be dealt with properly...
    cb_stats['geno_prob'] = cb_stats['geno_prob'].fillna(1.0)
    if cb_filter_exprs is not None:
        cb_stats = cb_stats.query(cb_filter_exprs)
    return cb_stats
