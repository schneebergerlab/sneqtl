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
                             rng=None):
    if rng is None:
        random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)
    else:
        random_state = np.random.RandomState(rng.integers(1000))
    if principal_components.empty:
        return None
    max_s = 0
    if n_clusters is None:
        for n in range(2, auto_max_clusters + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                random_state=random_state
            ).fit(principal_components)
            s = silhouette_score(principal_components, gmm.predict(principal_components))
            if s > max_s:
                max_s = s
                best_gmm = gmm
    else:
        best_gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            covariance_type=covariance_type,
        ).fit(principal_components)
        max_s = silhouette_score(principal_components, best_gmm.predict(principal_components))
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