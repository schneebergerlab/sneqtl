import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from snco.defaults import DEFAULT_RANDOM_SEED

log = logging.getLogger('sneqtl')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)

def filter_haplotype_correlated_principal_components(principal_components, haplotypes,
                                                     max_var_explained=0.05):
    n_pcs = principal_components.shape[1]
    keep_pcs = []
    for pc_idx, pc in principal_components.items():
        r2 = haplotypes.corrwith(pc, axis=0) ** 2
        max_r2 = r2.max()
        if max_r2 <= max_var_explained:
            keep_pcs.append(pc_idx)
        else:
            chrom, pos = r2.idxmax()
            log.info(
                f'Removing {pc_idx} as it correlates with '
                f'{chrom}:{pos/1e6:.1f} '
                f'with R squared of {max_r2:.2f}'
            )
    return keep_pcs


def get_principal_components(exprs_mat, haplotypes,
                             min_var_explained=0.01, max_pcs=10,
                             max_pc_haplotype_var_explained=0.05,
                             max_fit_barcodes=10_000,
                             return_pca_transformer=False,
                             rng=DEFAULT_RNG):
    if rng is None:
        random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    else:
        random_state = np.random.RandomState(rng.integers(1000))
    if exprs_mat.shape[1] > max_fit_barcodes:
        sample_barcodes = rng.choice(exprs_mat.columns.values, size=max_fit_barcodes, replace=False)
    else:
        sample_barcodes = exprs_mat.columns.values
    pca = PCA(n_components=max_pcs, svd_solver='arpack', random_state=random_state)
    pca = pca.fit(exprs_mat.T.loc[sample_barcodes])
    principal_components = pca.transform(exprs_mat.T)
    stop_idx = max_pcs - np.searchsorted(
        pca.explained_variance_ratio_[::-1],
        min_var_explained
    )

    pc_columns = [f'PC{i}' for i in range(1, stop_idx + 1)]
    principal_components = pd.DataFrame(
        principal_components[:, :stop_idx],
        columns=pc_columns,
        index=exprs_mat.columns
    )
    tot_ve = pca.explained_variance_ratio_[:stop_idx].sum() * 100
    log.info(f'Identified {stop_idx} principal components explaining {tot_ve:.2f}% of variance')
    filtered_pc_columns = filter_haplotype_correlated_principal_components(
        principal_components.loc[sample_barcodes],
        haplotypes.loc[sample_barcodes],
        max_pc_haplotype_var_explained
    )

    if not return_pca_transformer:
        return principal_components[filtered_pc_columns]

    def _pca_transform(new_exprs_mat):
        new_principal_components = pd.DataFrame(
            pca.transform(new_exprs_mat.T)[:, :stop_idx],
            columns=pc_columns,
            index=new_exprs_mat.columns
        )
        return new_principal_components[filtered_pc_columns]

    return principal_components[filtered_pc_columns], _pca_transform