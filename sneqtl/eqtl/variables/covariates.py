import numpy as np
from .base import VariableManager
from . import utils


class Covariates(VariableManager):

    @property
    def variable_names(self):
        return self.columns
    
    @property
    def barcodes(self):
        return self.index

    @classmethod
    def from_variables(cls, *, interacting_variables=None, parental_genotypes=None,
                       control_haplotype_variables=None, base_covariates=None):
        columns = ['const',]
        indexes = []
        sections = []

        if interacting_variables is not None:
            iv = utils.get_numeric_variables(interacting_variables, drop_first=True)
            sections.append(iv._data)
            columns += iv.columns
            indexes.append(iv.index)

        if parental_genotypes is not None:
            pg = utils.get_numeric_variables(parental_genotypes)
            sections.append(pg._data)
            columns += pg.columns
            indexes.append(pg.index)

        if control_haplotype_variables is not None:
            sections.append(control_haplotype_variables._data)
            columns += control_haplotype_variables.columns
            indexes.append(control_haplotype_variables.index)

        if base_covariates is not None:
            bc = utils.get_numeric_variables(base_covariates, drop_first=True)
            sections.append(bc._data)
            columns += bc.columns
            indexes.append(bc.index)

        if not sections:
            raise ValueError('No covariates provided')

        index = indexes[0]
        if not all(np.array_equal(index, other_index) for other_index in indexes[1:]):
            raise ValueError('Indexes do not match')

        covariate_matrix = np.column_stack(sections)
        n_obs = covariate_matrix.shape[0]
        intercept = np.ones((n_obs, 1), dtype=covariate_matrix.dtype)
        covariate_matrix = np.column_stack([intercept, covariate_matrix])

        return cls(covariate_matrix, index=index, columns=columns)

    @classmethod
    def null_covariates(cls, variables, dtype=np.float32):
        n_obs = y.shape[0]
        intercept = np.ones((n_obs, 1), dtype=dtype)
        return cls(intercept, index=variables.index, columns=['const',])