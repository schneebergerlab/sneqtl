import numpy as np
from .base import VariableManager
from .utils import check_aligned


class Covariates(VariableManager):

    @property
    def variable_names(self):
        return self.columns
    
    @property
    def barcodes(self):
        return self.index

    @classmethod
    def from_variables(cls, *, interacting_variables=None, parental_genotypes=None,
                       covariate_haplotypes=None, base_covariates=None, name=None):
        columns = ['const',]
        sections = []

        index = check_aligned(
            *interacting_variables, parental_genotypes,
            covariate_haplotypes, base_covariates
        )

        if interacting_variables is None:
            interacting_variables = []
        for iv in interacting_variables:
            iv = iv.as_covariates()
            sections.append(iv._data)
            columns += iv.columns

        if parental_genotypes is not None:
            pg = parental_genotypes.as_covariates()
            sections.append(pg._data)
            columns += pg.columns

        if covariate_haplotypes is not None:
            sections.append(covariate_haplotypes._data)
            columns += covariate_haplotypes.columns

        if base_covariates is not None:
            sections.append(base_covariates._data)
            columns += base_covariates.columns

        if not sections:
            return None

        covariate_matrix = np.column_stack(sections)
        n_obs = covariate_matrix.shape[0]
        intercept = np.ones((n_obs, 1), dtype=covariate_matrix.dtype)
        covariate_matrix = np.column_stack([intercept, covariate_matrix])

        return cls(covariate_matrix, index=index, columns=columns, name=name)

    @classmethod
    def null_covariates(cls, variables, dtype=np.float32, name=None):
        n_obs = y.shape[0]
        intercept = np.ones((n_obs, 1), dtype=dtype)
        return cls(intercept, index=variables.index, columns=['const',], name=name)