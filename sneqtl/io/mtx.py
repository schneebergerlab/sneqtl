import os
import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite


def read_index(idx_fn):
    with open(idx_fn) as f:
        idx = [r.strip().split('\t')[0] for r in f.readlines()]
    return np.array(idx)


def read_matrix_market(mtx_dir,
                  mtx_fn='matrix.mtx',
                  barcode_fn='barcodes.tsv',
                  feature_fn='features.tsv'):
    features = read_index(os.path.join(mtx_dir, feature_fn))
    barcodes = read_index(os.path.join(mtx_dir, barcode_fn))
    mtx = mmread(os.path.join(mtx_dir, mtx_fn)).tocsr()
    return features, barcodes, mtx


def read_matrix_market_as_pd_matrix(mtx_dir, cb_whitelist=None,
                                    as_dense=True, dense_dtype=np.float32):
    features, barcodes, mtx = read_matrix_market(mtx_dir)
    mtx = pd.DataFrame.sparse.from_spmatrix(mtx, columns=barcodes, index=features)
    if cb_whitelist is not None:
        mtx = mtx.loc[:, mtx.columns.isin(cb_whitelist)]
    if as_dense:
        mtx = mtx.sparse.to_dense().astype(dense_dtype)
    return mtx


def lognormalise_exprs(mtx, alpha=0.25):
    umi_counts = mtx.sum(0)
    sf = (umi_counts / np.mean(umi_counts))
    ln_mtx = np.log2(mtx / sf + (1.0 / (4 * alpha)))
    ln_mtx = ln_mtx.replace([np.inf, -np.inf], np.nan).fillna(0)
    sf = sf.rename('size_factors')
    return ln_mtx, sf


def read_expression_matrix(exprs_mat_dir, cb_whitelist=None,
                           rel_min_cells_exprs=0.02,
                           abs_min_cells_exprs=100,
                           min_variance=1e-4):
    exprs_mat = read_matrix_market_as_pd_matrix(exprs_mat_dir, cb_whitelist, as_dense=True)
    exprs_mat, sf = lognormalise_exprs(exprs_mat)
    min_cells_exprs = max(exprs_mat.shape[1] * rel_min_cells_exprs, abs_min_cells_exprs)
    exprs_mat = exprs_mat[(exprs_mat > 0).sum(axis=1) > min_cells_exprs]

    gene_vars = exprs_mat.var(axis=1)
    exprs_mat = exprs_mat.loc[gene_vars > min_variance]
    return exprs_mat


def write_features(feats_list, output_fn):
    with open(output_fn, 'w') as f:
        for feat in feats_list:
            f.write(f'{feat}\n')


def write_matrix_market(output_dir, gene_ids, cb_whitelist, mtx):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cb_fn = os.path.join(output_dir, 'barcodes.tsv')
    feat_fn = os.path.join(output_dir, 'features.tsv')
    mtx_fn = os.path.join(output_dir, 'matrix.mtx')
    write_features(cb_whitelist, cb_fn)
    write_features(gene_ids, feat_fn)
    mmwrite(mtx_fn, mtx)
