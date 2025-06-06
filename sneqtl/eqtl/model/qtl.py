from functools import cached_property
import numpy as np

from .lm import fit_model, compare_lr_test, parameter_stats
from .lm_torch import fit_models_torch, permuted_residuals, TORCH_AVAILABLE
from sneqtl.variables.covariates import Covariates
from sneqtl.variables.results import SingleGeneResults


DEFAULT_RANDOM_SEED = 101
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


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

    def permutation_lod_thresholds(self, n_permutations=50, n_loci_per_permutation=200,
                                   alpha=0.01, rng=DEFAULT_RNG):
        if not TORCH_AVAILABLE or fit_models_torch is None:
            raise RuntimeError("Permuation QTL testing is only available with pytorch")

        n_loci = self.haplotype_variables._data.shape[1]
        n_loci_per_permutation = min(n_loci_per_permutation, n_loci)
        n_params = self.n_hap_vars
        

        lrt_lods = []
        max_param_lods = []

        # 
        y_permutation_iter = permuted_residuals(
            self.y, self.covariates._data,
            rng=rng,
            n_permutations=n_permutations
        )

        for y_perm in y_permutation_iter:

            loci_idx = rng.choice(n_loci, size=n_loci_per_permutation, replace=False)

            result = fit_models_torch(
                y_perm,
                self.covariates._data,
                self.haplotype_variables._data[:, loci_idx, :],
                batch_size=n_loci_per_permutation,
                mode='lod_only'
            )

            # First column: full model LOD
            lrt_lods.extend(result[:, 0])

            # Next n_params columns: per-parameter LOD scores
            param_lods = result[0, 2:2 + n_params]
            max_param_lods.extend(np.max(result[:, 1:1 + n_params], axis=1))

        # Empirical thresholds at (1 - alpha) quantile
        lrt_lod_thresh = np.percentile(lrt_lods, 100 * (1 - alpha))
        param_lod_thresh = np.percentile(max_param_lods, 100 * (1 - alpha))

        return lrt_lod_thresh, param_lod_thresh

    def test_all_positions(self, use_torch=True, use_permutation_testing=True,
                           n_permutations=50, n_loci_per_permutation=200,
                           alpha=0.01, rng=DEFAULT_RNG):
        if use_torch and TORCH_AVAILABLE and fit_models_torch is not None:

            res = fit_models_torch(
                self.y,
                self.covariates._data,
                self.haplotype_variables._data,
                mode='full',
            )
            res = SingleGeneResults(
                res,
                index=self.haplotype_variables.haplotypes,
                columns=self.result_colnames,
                name=self.pheno_name,
            )
            
            if use_permutation_testing:
                lrt_lod_thresh, param_lod_thresh = self.permutation_lod_thresholds(
                    n_permutations=n_permutations, n_loci_per_permutation=n_loci_per_permutation,
                    alpha=alpha, rng=rng
                )
                res = res.apply_lod_thresholds(lrt_lod_thresh, param_lod_thresh)

            return res

        res = []
        for (chrom, pos), haplo in self.haplotype_variables.iter_haplotypes():
            res.append(self.run_test(haplo))
        return SingleGeneResults(
            np.array(res),
            index=self.haplotype_variables.haplotypes,
            columns=self.result_colnames,
            name=self.pheno_name
        )