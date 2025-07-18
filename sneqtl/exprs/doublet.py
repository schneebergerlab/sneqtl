import logging
import numpy as np
import pandas as pd

from snco.predict.doublet import k_nearest_neighbours_classifier
from snco.defaults import DEFAULT_RANDOM_SEED
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.variables.pca import PrincipalComponents
from sneqtl.variables.celltypes import CellTypes


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
    return ExpressionVariables(
        syn_doublets,
        index=exprs_mat.index,
        columns=np.arange(n_sim),
    )


def predict_expression_doublets(exprs_mat, principal_components, celltype_clusters,
                                n_doublets, k_neighbours, multiplet_factor=0.25,
                                rng=DEFAULT_RNG):

    min_cluster_size = min(celltype_clusters.celltype_counts.values())

    if n_doublets > 1:
        n_sim = int(min(n_doublets, min_cluster_size))
    else:
        if not n_doublets:
            raise ValueError('n-doublets must be >0 for doublet detection')
        n_sim = int(min_cluster_size * n_doublets)

    syn_doublet_exprs_mat = simulate_expression_doublets(
        exprs_mat, celltype_clusters, n_sim,
        multiplet_factor=multiplet_factor,
    )
    syn_doublet_principal_components = principal_components.transform(syn_doublet_exprs_mat)
    X_train = np.concatenate(
        [
            principal_components.iloc[np.random.randint(0, len(principal_components.barcodes), size=n_sim)],
            syn_doublet_principal_components._data
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
        knn_classifier(principal_components._data),
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