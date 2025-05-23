import pandas as pd
from functools import cached_property

from .lm import fit_model, compare_lr_test, parameter_stats
from ..variables.haplotypes import HaplotypeVariables
from ..variables.covariates import Covariates


class SinglePhenoQTLAnalysis:

    def __init__(self, y, haplotype_variables, covariates=None, pheno_name=None):
        self.y = y
        self.pheno_name = pheno_name
        self.haplotype_variables = haplotype_variables
        self.n_hap_vars = len(self.haplotype_variables.variable_names)
        if covariates is None:
            self.covariates = Covariates.null_covariates(haplotype_variables)
        else:
            self.covariates = covariates

        m_nest = fit_model(self.y, self.covariates._data)
        self.nested_llf = m_nest.llf
        self.nested_df = m_nest.df_model

    @cached_property
    def result_colnames(self):
        variable_names = self.haplotype_variables.variable_names
        return (
            'chrom', 'pos', 'phenotype',
            'lod_score', 'lrt_pval',
            *(f'{var_name}_lod_score' for var_name in variable_names),
            *(f'{var_name}_pval' for var_name in variable_names),
            *(f'{var_name}_coef' for var_name in variable_names),
        )

    def run_test(self, haplo):
        X = np.column_stack([haplo, self.covariates._data])
        m_full = fit_model(self.y, X)
        lod_score, pval = compare_lr_test(m_full, self.nested_llf, self.nested_df)
        param_lods, param_pvals, param_coefs = parameter_stats(m_full, n_tests=self.n_hap_vars)
        return (lod_score, pval, *param_lods, *param_pvals, *param_coefs)

    def test_position(self, chrom, pos):
        haplo = self.haplotype_variables[(chrom, pos)]
        return self.run_test(haplo)

    def test_all_positions(self):
        results = []
        pheno_name = self.pheno_name
        for (chrom, pos), haplo in self.haplotype_variables.iter_haplotypes():
            results.append(
                (chrom, pos, pheno_name, *self.run_test(haplo))
            )
        return pd.DataFrame(results, columns=self.result_colnames)
