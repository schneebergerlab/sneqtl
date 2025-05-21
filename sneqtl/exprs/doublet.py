import logging
import numpy as np
import pandas as pd

from ..covars import get_principal_components, create_celltype_clusters
from snco.predict.doublet import k_nearest_neighbours_classifier
from snco.defaults import DEFAULT_RANDOM_SEED


log = logging.getLogger('sneqtl')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def simulate_expression_doublets(exprs_mat, celltype_clusters, n_sim,
                                 multiplet_factor=0.25,
                                 rng=DEFAULT_RNG):
    celltype_clusters = celltype_clusters.values.argmax(axis=1)
    cluster_labels, cluster_counts = np.unique(celltype_clusters, return_counts=True)
    n_clusters = len(cluster_labels)
    n_genes = len(exprs_mat)
    syn_doublets = []
    for i in range(n_clusters):
        cluster_i_idx = np.argwhere(celltype_clusters == i).squeeze()
        syn_idx = rng.choice(cluster_i_idx, size=n_sim, replace=True)
        syn_doublets.append(exprs_mat.iloc[:, syn_idx])
    syn_doublets = np.array(syn_doublets)
    barcode_weights = rng.poisson(multiplet_factor, size=(n_clusters, n_sim)) + 1
    barcode_weights = np.tile(barcode_weights, n_genes).reshape(n_clusters, n_genes, n_sim)
    syn_doublets = np.average(syn_doublets, axis=0, weights=barcode_weights)
    return pd.DataFrame(
        syn_doublets,
        index=exprs_mat.index,
    )


def predict_expression_doublets(exprs_mat, haplotypes, n_doublets, k_neighbours,
                                min_var_explained=0.01, max_pcs=50,
                                max_pc_haplotype_var_explained=0.05,
                                n_clusters=None, auto_max_clusters=10,
                                covariance_type='spherical',
                                multiplet_factor=0.25,
                                rng=DEFAULT_RNG):
    principal_components, pca_transform = get_principal_components(
        exprs_mat, haplotypes,
        min_var_explained=min_var_explained,
        max_pcs=max_pcs,
        max_pc_haplotype_var_explained=max_pc_haplotype_var_explained,
        return_pca_transformer=True,
        rng=rng,
    )
    celltype_clusters_init = create_celltype_clusters(
        principal_components,
        n_clusters=n_clusters,
        auto_max_clusters=auto_max_clusters,
        covariance_type=covariance_type,
        rng=rng
    )
    celltype_cluster_counts = celltype_clusters_init.idxmax(axis=1).value_counts()
    min_cluster_size = celltype_cluster_counts.min()

    if n_doublets > 1:
        n_sim = int(min(n_doublets, min_cluster_size))
    else:
        if not n_doublets:
            raise ValueError('n-doublets must be >0 for doublet detection')
        n_sim = int(min_cluster_size * n_doublets)

    syn_doublet_exprs_mat = simulate_expression_doublets(
        exprs_mat, celltype_clusters_init, n_sim,
        multiplet_factor=multiplet_factor,
    )
    syn_doublet_principal_components = pca_transform(syn_doublet_exprs_mat)
    X_train = np.concatenate(
        [
            principal_components.sample(n=n_sim, replace=True, random_state=rng).values,
            syn_doublet_principal_components.values
        ],
        axis=0,
    )
    y_train = np.repeat([0, 1], [n_sim, n_sim])

    if k_neighbours > 1:
        k_neighbours = int(min(k_neighbours, n_sim))
    else:
        k_neighbours = int(n_sim * k_neighbours)

    log.info(f'Predicting doublets from expression data using {n_sim} synthetic doublets')
    knn_classifier = k_nearest_neighbours_classifier(X_train, y_train, k_neighbours)

    doublet_pred = pd.Series(
        knn_classifier(principal_components),
        index=principal_components.index,
        name='doublet_prediction',
    )
    doublet_n = (doublet_pred > 0.5).sum()
    log.info(
        f'Identified {doublet_n} putative doublets ({doublet_n / len(doublet_pred) * 100:.2f}%)'
    )
    if log.getEffectiveLevel() <= logging.DEBUG:
        X_pred_series = pd.Series(
            knn_classifier(X_train) > 0.5, name='Prediction'
        ).map({False: 'hq', True: 'doublet'})
        y_pred_series = pd.Series(y_train, name='Simulation').map({0: 'real', 1: 'sim'})
        log.debug(pd.crosstab(X_pred_series, y_pred_series))
    return doublet_pred