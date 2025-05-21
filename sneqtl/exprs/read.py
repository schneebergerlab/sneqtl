import os
import numpy as np
import pandas as pd
from scipy.io import mmread


def read_index(idx_fn):
    with open(idx_fn) as f:
        idx = [r.strip().split('\t')[0] for r in f.readlines()]
    return np.array(idx)


def read_mmformat(mtx_dir,
                  mtx_fn='matrix.mtx',
                  barcode_fn='barcodes.tsv',
                  feature_fn='features.tsv'):
    features = read_index(os.path.join(mtx_dir, feature_fn))
    barcodes = read_index(os.path.join(mtx_dir, barcode_fn))
    mtx = mmread(os.path.join(mtx_dir, mtx_fn)).tocsr()
    return features, barcodes, mtx


def read_mmformat_as_pd_matrix(mtx_dir, cb_whitelist=None, feat_filt=10,
                               as_dense=True, dtype=np.float32):
    features, barcodes, mtx = read_mmformat(mtx_dir)
    mtx = pd.DataFrame.sparse.from_spmatrix(mtx, columns=barcodes, index=features)
    if cb_whitelist is not None:
        mtx = mtx.loc[:, mtx.columns.isin(cb_whitelist)]
    if feat_filt is not None:
        mtx = mtx.loc[np.count_nonzero(mtx, axis=1) >= feat_filt]
    if as_dense:
        mtx = mtx.sparse.to_dense()
    mtx = mtx.astype(dtype)
    return mtx


def lognormalise_exprs(mtx, alpha=0.25):
    umi_counts = mtx.sum(0)
    sf = (umi_counts / np.mean(umi_counts))
    ln_mtx = np.log2(mtx / sf + (1.0 / (4 * alpha)))
    sf = sf.rename('size_factors')
    return ln_mtx, sf


def read_expression_matrix(exprs_mat_dir, cb_whitelist,
                           rel_min_cells_exprs=0.02,
                           abs_min_cells_exprs=100):
    exprs_mat = read_mmformat(exprs_mat_dir, cb_whitelist,).astype(np.float32)
    exprs_mat, sf = lognormalise_exprs(exprs_mat)
    min_cells_exprs = max(exprs_mat.shape[1] * rel_min_cells_exprs, abs_min_cells_exprs)
    exprs_mat = exprs_mat[(exprs_mat > 0).sum(axis=1) > min_cells_exprs]
    return exprs_mat