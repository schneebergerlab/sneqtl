import logging

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from snco.defaults import DEFAULT_RANDOM_SEED

log = logging.getLogger('sneqtl')


def create_celltype_clusters(principal_components, n_clusters=None,
                             auto_max_clusters=10,
                             covariance_type='spherical',
                             max_fit_barcodes=10_000,
                             rng=None):
    if rng is None:
        random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)
    else:
        random_state = np.random.RandomState(rng.integers(1000))
    if principal_components.empty:
        return None

    if principal_components.shape[0] > max_fit_barcodes:
        sample_barcodes = rng.choice(principal_components.index.values, size=max_fit_barcodes, replace=False)
    else:
        sample_barcodes = principal_components.index.values

    training_sample_pcs = principal_components.loc[sample_barcodes]

    max_s = 0
    if n_clusters is None:
        for n in range(2, auto_max_clusters + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                random_state=random_state
            ).fit(training_sample_pcs)
            s = silhouette_score(training_sample_pcs, gmm.predict(training_sample_pcs))
            if s > max_s:
                max_s = s
                best_gmm = gmm
    else:
        best_gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            covariance_type=covariance_type,
        ).fit(training_sample_pcs)
        max_s = silhouette_score(training_sample_pcs, best_gmm.predict(training_sample_pcs))
    log.info(f'Identified {best_gmm.n_components} putative '
             f'cell-type clusters with silhouette score {max_s:.2f}')
    cluster_preds = best_gmm.predict_proba(principal_components)
    _, cluster_counts = np.unique(cluster_preds.argmax(axis=1), return_counts=True)
    cluster_preds = cluster_preds[:, np.argsort(cluster_counts)[::-1]]
    log.debug(f'Cell-type cluster sizes: {sorted(cluster_counts)}')
    celltype_clusters = pd.DataFrame(
        cluster_preds,
        index=principal_components.index,
        columns=[f'celltype{i}' for i in range(1, best_gmm.n_components + 1)]
    )
    return celltype_clusters