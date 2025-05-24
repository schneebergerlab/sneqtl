import os
import tempfile

import numpy as np
import joblib
from snco.logger import progress_bar
from .qtl import SinglePhenoQTLAnalysis

from ..variables.exprs import ExpressionVariables
from ..variables.covariates import Covariates
from ..variables.interactions import HaplotypeInteractionVariables
from ..variables.utils import check_indexes_equivalent

class eQTLAnalysisManager:

    def __init__(self, expression_data, haplotypes,
                 parental_genotypes=None, interacting_variables=None,
                 covariates=None, covariate_haplotypes=None):

        check_indexes_equivalent(
            expression_data, haplotypes, parental_genotypes,
            interacting_variables, covariates
        )

        self._memmap_path = tempfile.mksdir()
        self._memmap_prefixes = {
            'exprs_prefix': os.path.join(self._memmap_path, 'exprs'),
            'covar_prefix': os.path.join(self._memmap_path, 'covar'),
            'haplo_prefix': os.path.join(self._memmap_path, 'haplo'),
        }

        if covariate_haplotypes is not None:
            haplotypes, covariate_haplotypes = haplotypes.get_control_haplotypes(
                control_haplotypes,
                remove_correlated=True
            )
        self.expression_data = expression.memmap(self._memmap_prefixes['exprs_prefix'])
        self.haplotype_variables = HaplotypeInteractionVariables.from_cross_product(
            haplotypes=haplotypes,
            parental_genotypes=parental_genotypes,
            interacting_variables=interacting_variables,
            memmap=memmap,
            memmap_path_prefix=self._memmap_prefixes['haplo_prefix']
        )
        self.covariates = Covariates(
            parental_genotypes=parental_genotypes,
            interacting_variables=interacting_variables,
            covariate_haplotypes=covariate_haplotypes,
            base_covariates=covariates,
        ).memmap(self._memmap_prefixes['covar_prefix'])


    def run_gene(self, gene_id):
        e = SinglePhenoQTLAnalysis(
            y=self.expression_data.loc[gene_id],
            haplotype_variables=self.haplotype_variables,
            covariates=self.covariates
        )
        return e.test_all_positions()

    def run_all_genes(self, processes=1):

        with tempfile.TemporaryDirectory() as o_dir:
    
            def _run_gene_parallel(gene_id, mmap_prefixes):
                expression_data = ExpressionVariables.load_memmap(
                    mmap_prefixes['exprs_prefix']
                )
                haplotype_variables = HaplotypeInteractionVariables.load_memmap(
                    mmap_prefixes['haplo_prefix']
                )
                covariates = Covariates.load_memmap(mmap_prefixes['covar_prefix'])
                e = SinglePhenoQTLAnalysis(
                    y=expression_data.loc[gene_id],
                    haplotype_variables=haplotype_variables,
                    covariates=covariates
                )
                res = e.test_all_positions()
                joblib.dump(res, os.path.join(o_dir, f'{gene_id}.pickle'))


            with joblib.Parallel(n_jobs=processes, backend='loky') as pool:
                gene_progress = progress_bar(
                    self.expression_data.gene_ids,
                    label='Running eQTL',
                    item_show_func=str,
                )
                with gene_progress:
                    pool(
                        joblib.delayed(_run_gene_parallel)(gene_id, self._memmap_prefixes)
                        for gene_id in gene_progress
                    )
            results = pd.concat((
                joblib.load(os.path.join(o_dir, f'{gene_id}.pickle'))
                for gene_id in self.expression_data.gene_ids
            ), axis=0, ignore_index=True)
            return results