import os
import shutil
import tempfile
import numpy as np
import joblib
import pandas as pd

from snco.logger import progress_bar
from .qtl import SinglePhenoQTLAnalysis
from .memmap import MemmapManager

from ..variables.base import VariableManager
from ..variables.exprs import ExpressionVariables
from ..variables.covariates import Covariates
from ..variables.interactions import HaplotypeInteractionVariables
from ..variables.results import eQTLResults
from ..variables.utils import check_indexes_equivalent


class eQTLAnalysisManager:

    def __init__(self, expression_variables, haplotypes,
                 parental_genotypes=None, interacting_variables=None,
                 covariates=None, covariate_haplotypes=None,
                 memmap=True):

        check_indexes_equivalent(
            expression_variables, haplotypes, parental_genotypes,
            interacting_variables, covariates
        )
        self._mm = MemmapManager(self, memmap)
        self.expression_variables = self._mm.convert(
            expression_variables, 'expression_variables'
        )

        self.covariates = self._mm.convert(covariates, 'covariates')

        self.haplotype_variables = HaplotypeInteractionVariables.from_cross_product(
            haplotype_variables=haplotypes,
            parental_genotypes=parental_genotypes,
            interacting_variables=interacting_variables,
            memmap_manager=self._mm,
            memmap_dataset_name='haplotype_variables',
        )
        self.results = None

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

    def run_all_genes(self, processes=1):

        gene_ids = self.expression_variables.gene_ids
        haplotypes = self.haplotype_variables.haplotypes
        variable_names = self.haplotype_variables.variable_names
        results_colnames = return (
            'lod_score', 'lrt_pval',
            *(f'{var_name}_lod_score' for var_name in variable_names),
            *(f'{var_name}_pval' for var_name in variable_names),
            *(f'{var_name}_coef' for var_name in variable_names),
        )
        self.results = self._mm.create_empty(
            'results',
            index=gene_ids,
            columns=haplotypes,
            items=results_colnames,
            dtype=np.float64,
            cls=eQTLResults,
        )

        def _run_gene_parallel(gene_id, mmap_prefixes):
            expression_variables = ExpressionVariables.load_memmap(
                mmap_prefixes['expression_variables']
            )
            haplotype_variables = HaplotypeInteractionVariables.load_memmap(
                mmap_prefixes['haplotype_variables']
            )
            covariates = Covariates.load_memmap(mmap_prefixes['covariates'])
            results = eQTLResults.load_memmap(mmap_prefixes['results'])

            e = SinglePhenoQTLAnalysis(
                y=expression_variables.loc[gene_id],
                haplotype_variables=haplotype_variables,
                covariates=covariates
            )
            res = e.test_all_positions()
            results.loc[gene_id, ...] = res._data

        gene_progress = progress_bar(
            self.expression_variables.gene_ids,
            label='Running eQTL',
            item_show_func=str,
        )
        with gene_progress:
            joblib.Parallel(n_jobs=processes, backend='loky')(
                joblib.delayed(_run_gene_parallel)(gene_id, self._mm.memmap_prefixes)
                for gene_id in gene_progress
            )

        return self.results
