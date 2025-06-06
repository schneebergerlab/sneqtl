import numpy as np
from .base import VariableManager
from .var_utils import check_aligned


class Covariates(VariableManager):

    @property
    def variable_names(self):
        return self.columns
    
    @property
    def barcodes(self):
        return self.index

    @classmethod
    def from_variables(cls, *, interacting_variables=None, parental_genotypes=None,
                       covariate_haplotypes=None, base_covariates=None,
                       parental_genotypes_covar_type='diploid', name=None):
        columns = ['const',]
        sections = []

        if interacting_variables is None:
            interacting_variables = []

        index = check_aligned(
            *interacting_variables, parental_genotypes,
            covariate_haplotypes, base_covariates
        )

        for iv in interacting_variables:
            iv = iv.as_covariates()
            sections.append(iv._data)
            columns += iv.columns

        if parental_genotypes is not None:
            if parental_genotypes_covar_type == 'diploid':
                pg = parental_genotypes.diploid_covariates
            else:
                pg = parental_genotypes.haploid_covariates
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

        covars = cls(covariate_matrix, index=index, columns=columns, name=name)
        if covars.is_singular:
            problem_covar = covars.find_dependent_variables()
            raise ValueError(
                f'Covariate matrix is singular - variables {problem_covar} are dependent (there may be others)'
            )
        return covars

    @classmethod
    def null_covariates(cls, variables, dtype=np.float32, name=None):
        n_obs = variables.shape[0]
        intercept = np.ones((n_obs, 1), dtype=dtype)
        return cls(intercept, index=variables.index, columns=['const',], name=name)