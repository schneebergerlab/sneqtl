from functools import cached_property
import numpy as np

from .lm import fit_models
from sneqtl.variables.covariates import Covariates
from sneqtl.variables.results import SingleGeneResults


DEFAULT_RANDOM_SEED = 101
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


class SinglePhenoQTLAnalysis:

    def __init__(self, y, haplotype_variables, covariates=None,
                 haplotype_variables_alt=None, pheno_name=None):
        self.y = y
        self.pheno_name = pheno_name
        self.haplotype_variables = haplotype_variables
        self.haplotype_variables_alt = haplotype_variables_alt
        if covariates is None:
            self.covariates = Covariates.null_covariates(haplotype_variables)
        else:
            self.covariates = covariates

    @cached_property
    def result_colnames(self):
        variable_names = self.haplotype_variables.variable_names
        results_colnames = (
            'lod_score', 'lrt_pval',
            *(f'{var_name}_lod_score' for var_name in variable_names),
            *(f'{var_name}_pval' for var_name in variable_names),
            *(f'{var_name}_coef' for var_name in variable_names),
        )
        if self.haplotype_variables_alt is not None:
            variable_names_alt = self.haplotype_variables_alt.variable_names
            results_colnames = results_colnames + (
                'lod_score_alt', 'lrt_pval_alt',
                *(f'{var_name}_lod_score' for var_name in variable_names_alt),
                *(f'{var_name}_pval' for var_name in variable_names_alt),
                *(f'{var_name}_coef' for var_name in variable_names_alt),
            )
        return results_colnames

    def test_all_positions(self, n_perm=50, n_haps_per_perm=200,
                           alpha=0.01, rng=DEFAULT_RNG):

        res = fit_models_torch(
            self.y,
            self.covariates._data,
            self.haplotype_variables._data,
            self.haplotype_variables_alt._data,
            n_perm=n_perm,
            n_haps_per_perm=n_haps_per_perm,
            alpha=alpha
            rng=rng
        )
        res = SingleGeneResults(
            res,
            index=self.haplotype_variables.haplotypes,
            columns=self.result_colnames,
            name=self.pheno_name,
        )
        
        return res
