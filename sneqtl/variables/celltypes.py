import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from .base import VariableManager

DEFAULT_RANDOM_SEED = 101
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


class CellTypes(VariableManager):

    @property
    def barcodes(self):
        return self.index

    @property
    def celltypes(self):
        return self.columns

    def as_covariates(self):
        return self.reindex(columns=list(self.columns[1:]))

    @property
    def celltype_counts(self):
        counts = np.bincount(np.argmax(self._data, axis=1), minlength=self.shape[1])
        celltypes = self.columns
        return {
            k: v for k, v in zip(celltypes, counts)
        }

    @classmethod
    def from_principal_components(cls, principal_components,
                                  n_clusters=None, auto_max_clusters=10,
                                  covariance_type='spherical',
                                  max_fit_barcodes=10_000,
                                  rng=DEFAULT_RNG):
        if rng is None:
            random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)
            rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
        else:
            random_state = np.random.RandomState(rng.integers(1000))
        if principal_components is None or not principal_components.size:
            return None

        if principal_components.shape[0] > max_fit_barcodes:
            sample_barcodes = rng.choice(principal_components.index, size=max_fit_barcodes, replace=False)
        else:
            sample_barcodes = list(principal_components.index)

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

        preds = best_gmm.predict_proba(principal_components._data)
        _, counts = np.unique(preds.argmax(axis=1), return_counts=True)
        preds = preds[:, np.argsort(counts)[::-1]]

        columns = [f'celltype{i}' for i in range(1, best_gmm.n_components + 1)]
        return cls(preds, index=principal_components.index, columns=columns)
