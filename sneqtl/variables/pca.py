import os
import numpy as np
from sklearn.decomposition import PCA

from .base import VariableManager

DEFAULT_RANDOM_SEED = 101
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


class PrincipalComponents(VariableManager):

    index_name = 'cb'
    columns_name = 'principal_components'

    def __init__(self, data, index, columns, items=None, name=None, dtype=None, frozen=True):
        super().__init__(data, index, columns, items=None, name=None, dtype=None, frozen=True)
        self.pc_type = None
        self._pca = None
        self.var_explained = None

    @property
    def barcodes(self):
        return self.index

    @property
    def components(self):
        return self.columns

    def transform(self, new_expression_variables, name=None):
        if self._pca is None:
            raise ValueError('PCA transformer not available')
        transformed = self._pca.transform(new_expression_variables.T._data)
        return self.__class__(
            transformed[:, :len(self.components)],
            index=new_expression_variables.columns,
            columns=self.components,
            name=name,
            dtype=transformed.dtype
        )

    @classmethod
    def from_expression(cls, expression_variables,
                        min_var_explained=0.01, max_pcs=10,
                        max_fit_barcodes=10_000, name=None,
                        rng=DEFAULT_RNG):
        if rng is None:
            random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)
            rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
        else:
            random_state = np.random.RandomState(rng.integers(1000))
        if expression_variables.shape[1] > max_fit_barcodes:
            sample_barcodes = rng.choice(expression_variables.columns, size=max_fit_barcodes, replace=False)
        else:
            sample_barcodes = list(expression_variables.columns)
        pca = PCA(n_components=max_pcs, svd_solver='arpack', random_state=random_state)
        pca = pca.fit(expression_variables.T.loc[sample_barcodes])
        principal_components = pca.transform(expression_variables.T._data)
        stop_idx = max_pcs - np.searchsorted(
            pca.explained_variance_ratio_[::-1],
            min_var_explained
        )

        pc_columns = [f'ePC{i}' for i in range(1, stop_idx + 1)]
        principal_components = cls(
            principal_components[:, :stop_idx],
            index=expression_variables.columns,
            columns=pc_columns,
            name=name,
            dtype=principal_components.dtype
        )
        principal_components.pc_type = 'exprs'
        principal_components._pca = pca
        principal_components.var_explained = sum(pca.explained_variance_ratio_[:stop_idx])
        return principal_components

    @classmethod
    def from_variants(cls, variants, parental_genotypes=None,
                      min_var_explained=0.01, max_pcs=None,
                      name=None, rng=DEFAULT_RNG):
        if rng is None:
            random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)
            rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
        else:
            random_state = np.random.RandomState(rng.integers(1000))
        design, inverse = np.unique(
            parental_genotypes._hap1._data + parental_genotypes._hap2._data,
            axis=0,
            return_inverse=True
        )
        if len(design) < 2:
            raise ValueError(
                "No variation in parental founder dosages; "
                "variant-based PCA cannot be performed. "
                "(Note that reciprocal crosses are equivalent here)"
            )
        parental_variants = design @ variants._data.T
        max_pcs = min(
            max_pcs if max_pcs is not None else len(design) - 1,
            min(parental_variants.shape) - 1,
        )
        pca = PCA(n_components=max_pcs, svd_solver='arpack', random_state=random_state)
        pca = pca.fit(parental_variants)
        principal_components = pca.transform(parental_variants)
        stop_idx = max_pcs - np.searchsorted(
            pca.explained_variance_ratio_[::-1],
            min_var_explained
        )

        pc_columns = [f'vPC{i}' for i in range(1, stop_idx + 1)]
        principal_components = cls(
            principal_components[inverse, :stop_idx],
            index=parental_genotypes.index,
            columns=pc_columns,
            name=name,
            dtype=principal_components.dtype
        )
        principal_components.pc_type = 'variants'
        principal_components._pca = pca
        principal_components.var_explained = sum(pca.explained_variance_ratio_[:stop_idx])
        return principal_components

    @classmethod
    def from_local_variants(cls, local_variants, n_pcs,
                            name=None, rng=DEFAULT_RNG):
        if rng is None:
            random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)
            rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
        else:
            random_state = np.random.RandomState(rng.integers(1000))
        lv = local_variants._data.squeeze().T
        n_vars, n_haps = lv.shape
        if n_pcs > (min(n_vars, n_haps) - 1):
            raise ValueError('n_pcs must be smaller than n_vars, n_cb')
        pca = PCA(n_components=n_pcs, svd_solver='arpack', random_state=random_state)
        pca = pca.fit(lv)
        principal_components = pca.transform(lv)
        pc_columns = [f'cisvPC{i}' for i in range(1, n_pcs + 1)]
        principal_components = cls(
            principal_components,
            index=local_variants.columns,
            columns=pc_columns,
            name=name,
            dtype=principal_components.dtype
        )
        principal_components.pc_type = 'local_variants'
        principal_components._pca = pca
        principal_components.var_explained = sum(pca.explained_variance_ratio_)
        return principal_components