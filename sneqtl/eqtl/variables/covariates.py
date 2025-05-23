import numpy as np
from .variable_manager import VariableManager
from . import utils


class Covariates(VariableManager):

    @property
    def variable_names(self):
        return self.columns
    
    @property
    def barcodes(self):
        return self.index

    @classmethod
    def from_dataframes(cls, *, interacting_variables=None, parental_genotypes=None,
                        control_haplotype_variables=None, base_covariates=None):
        columns = ['const',]
        indexes = []
        sections = []

        if interacting_variables is not None and not interacting_variables.empty:
            iv = utils.convert_to_numeric(interacting_variables, drop_first=True)
            sections.append(iv.to_numpy())
            columns += iv.columns.tolist()
            indexes.append(iv.index.to_numpy())

        if parental_genotypes is not None and not parental_genotypes.empty:
            pg = utils.convert_to_numeric(parental_genotypes)
            sections.append(pg.to_numpy())
            columns += pg.columns.tolist()
            indexes.append(pg.index.to_numpy())

        if control_haplotype_variables is not None and not control_haplotype_variables.empty:
            sections.append(control_haplotype_variables.to_numpy())
            columns += control_haplotype_variables.columns.tolist()
            indexes.append(control_haplotype_variables.index.to_numpy())

        if base_covariates is not None:
            bc = utils.convert_to_numeric(base_covariates, drop_first=True)
            sections.append(bc.to_numpy())
            columns += bc.columns.tolist()
            indexes.append(bc.index.to_numpy())

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
