import os
import shutil
import tempfile
import numpy as np
import joblib
import pandas as pd
from scipy import stats

from snco.main.logger import progress_bar
from .model.qtl import SinglePhenoQTLAnalysis
from .model.lm_torch import TORCH_AVAILABLE
from .memmap import MemmapManager

from sneqtl.variables.base import VariableManager
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.variables.covariates import Covariates
from sneqtl.variables.interactions import HaplotypeInteractionVariables
from sneqtl.variables.results import eQTLResults, SingleGeneResults
from sneqtl.variables.var_utils import check_aligned


class eQTLAnalysisManager:

    def __init__(self, expression_variables, haplotypes,
                 parental_genotypes=None, parental_genotypes_covar_type='diploid',
                 interacting_variables=None, covariates=None,
                 covariate_haplotypes=None, use_permutation_testing=True):

        check_aligned(
            expression_variables.T, haplotypes, parental_genotypes,
            covariates, *interacting_variables
        )
        self._mm = MemmapManager(memmap=True)
        expression_variables.rename('expression_variables')
        self.expression_variables = self._mm.convert(expression_variables)
        covariate_haplotypes, haplotypes = haplotypes.get_covariate_haplotypes(
            covariate_haplotypes, remove_correlated=True,
        )
        covariates = Covariates.from_variables(
            parental_genotypes=parental_genotypes,
            interacting_variables=interacting_variables,
            covariate_haplotypes=covariate_haplotypes,
            base_covariates=covariates,
            parental_genotypes_covar_type='diploid',
            name='covariates'
        )
        self.covariates = self._mm.convert(covariates)

        self.h_eff = haplotypes.h_eff
        self.haplotype_variables = HaplotypeInteractionVariables.from_cross_product(
            haplotype_variables=haplotypes,
            parental_genotypes=parental_genotypes,
            interacting_variables=interacting_variables,
            name='haplotype_variables',
            memmap_manager=self._mm,
        )
        self.results = None
        self.use_permutation_testing = use_permutation_testing if TORCH_AVAILABLE else False

    @property
    def gene_ids(self):
        return self.expression_variables.gene_ids

    @property
    def haplotypes(self):
        return self.haplotype_variables.haplotypes

    @property
    def covariate_variable_names(self):
        return self.covariates.columns

    @property
    def haplotype_variable_names(self):
        return self.haplotype_variables.variable_names

    @property
    def results_colnames(self):
        variable_names = self.haplotype_variables.variable_names
        if self.use_permutation_testing:
            return ('lod_score', 'lrt_pval',
                *(f'{var_name}_lod_score' for var_name in variable_names),
                *(f'{var_name}_sig' for var_name in variable_names),
                *(f'{var_name}_pval' for var_name in variable_names),
                *(f'{var_name}_coef' for var_name in variable_names),
            )
        else:
            return ('lod_score', 'lrt_pval',
                *(f'{var_name}_lod_score' for var_name in variable_names),
                *(f'{var_name}_pval' for var_name in variable_names),
                *(f'{var_name}_coef' for var_name in variable_names),
            )

    def close(self):
        self._mm.remove_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def run_gene(self, gene_id):
        e = SinglePhenoQTLAnalysis(
            y=self.expression_variables.loc[gene_id],
            haplotype_variables=self.haplotype_variables,
            covariates=self.covariates
        )
        return e.test_all_positions()

    def apply_fwer_correction(self, results, inplace=True):
        if not inplace:
            results = results.copy()
        pval_cols = [p for p in results.results_colnames if p.endswith('pval')]
        results.loc[:, :, pval_cols] = np.clip(
            results.loc[:, :, pval_cols] * self.h_eff,
            a_max=1.0,
            a_min=np.finfo(np.float64).tiny
        )
        return results

    @staticmethod
    def apply_fdr_correction(results, inplace=True):
        if not inplace:
            results = results.copy()
        pval_cols = [p for p in results.results_colnames if p.endswith('pval')]
        for h in results.haplotypes:
            for pv in pval_cols:
                results.loc[:, h, pv] = stats.false_discovery_control(results.loc[:, h, pv])
        return results

    def run_all_genes(self, processes=1, correct_fwer=True, correct_fdr=True):

        def _run_gene_parallel(gene_id):
            expression_variables = self._mm.load('expression_variables')
            covariates = self._mm.load('covariates')
            haplotype_variables = self._mm.load('haplotype_variables')

            e = SinglePhenoQTLAnalysis(
                y=expression_variables.loc[gene_id],
                haplotype_variables=haplotype_variables,
                covariates=covariates,
                pheno_name=gene_id,
            )
            gene_results = e.test_all_positions()
            self._mm.convert(gene_results)

        gene_progress = progress_bar(
            self.gene_ids,
            label='Running eQTL',
            item_show_func=str,
        )
        with gene_progress:
            joblib.Parallel(n_jobs=processes, backend='loky')(
                joblib.delayed(_run_gene_parallel)(gene_id)
                for gene_id in gene_progress
            )

        results = self._mm.create_empty(
            'results',
            index=self.gene_ids,
            columns=self.haplotypes,
            items=self.results_colnames,
            dtype=np.float64,
            cls=eQTLResults,
        )
        results_arr = results._data
        for i, gene_id in enumerate(self.gene_ids):
            self._mm.register(gene_id, SingleGeneResults)
            gene_results = self._mm.load(gene_id, mode=None) # not memmapped
            results_arr[i, ...] = gene_results._data[:]
        if correct_fdr:
            results = self.apply_fdr_correction(results)
        if correct_fwer:
            results = self.apply_fwer_correction(results)
        return results
