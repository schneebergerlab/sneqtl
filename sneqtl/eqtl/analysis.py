import os
import shutil
import tempfile
import numpy as np
import joblib
import pandas as pd
from scipy import stats
import pyarrow as pa
from pyarrow import parquet as pq
from  pyarrow import dataset as ds

from coelsch.main.logger import progress_bar
from .model.qtl import SinglePhenoQTLAnalysis, SingleGeneResults
from .model.lm import precompute_var_covar_cholesky
from .memmap import MemmapManager

from sneqtl.variables.base import VariableManager
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.variables.covariates import Covariates
from sneqtl.variables.interactions import HaplotypeInteractionVariables
from sneqtl.variables.var_utils import check_aligned


class eQTLResults:
    def __init__(self, results_dir):
        self._dir = results_dir
        self.gene_ids = {
            os.path.splitext(e.name)[0]: e.path
            for e in os.scandir(results_dir)
            if e.is_file() and e.name.endswith(".parquet")
        }

        any_file = next(iter(self.gene_ids.values()))
        self.schema = pq.read_schema(any_file)
        self.bin_size = int(self.schema.field("pos").metadata[b"bin_size"].decode())

    @property
    def columns(self):
        return self.schema.names

    def __getitem__(self, gene_id):
        return SingleGeneResults.read_parquet(self.gene_ids[gene_id])

    def __len__(self):
        return len(self.gene_ids)

    def __iter__(self):
        return iter(self.gene_ids)

    def iter_results(self):
        for gene_id in self.gene_ids:
            yield gene_id, self[gene_id]


class eQTLAnalysisManager:

    def __init__(self, output_dir, expression_variables, haplotypes,
                 parental_genotypes=None, variants=None,
                 interacting_variables=None, covariates=None,
                 covariate_haplotypes=None,
                 genotype_covariates_as_pcs=False,
                 precompute_hap_factors=True,
                 minP_threshold=0.01,
                 n_perm=1000,
                 batch_size=500,
                 exists_ok=False):

        if interacting_variables is None:
            interacting_variables = []

        check_aligned(
            expression_variables.T, haplotypes, parental_genotypes,
            covariates, *interacting_variables
        )
        self._mm = MemmapManager(memmap=True)
        self.batch_size = batch_size
        self.minP_threshold = minP_threshold
        self.n_perm = n_perm

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
            variants=variants,
            genotype_covariates_as_pcs=genotype_covariates_as_pcs,
            name='covariates',
        )
        self.covariates = self._mm.convert(covariates)

        self.bin_size = haplotypes.bin_size

        self.haplotype_variables = HaplotypeInteractionVariables.from_cross_product(
            haplotype_variables=haplotypes,
            parental_genotypes=parental_genotypes,
            variant_variables=variants,
            interacting_variables=interacting_variables,
            name='haplotype_variables',
            memmap_manager=self._mm,
        )

        if precompute_hap_factors:
            self._precomputed_factors = True
            p = len(self.covariates.variable_names) + len(self.haplotype_variables.variable_names)
            self.L_precomp = self._mm.create_empty(
                'L_precomp',
                index=self.haplotype_variables.haplotypes,
                columns=[str(i) for i in range(p)],
                items=[str(i) for i in range(p)],
                dtype=np.float64,
            )
            self.diag_precomp = self._mm.create_empty(
                'diag_precomp',
                index=self.haplotype_variables.haplotypes,
                columns=[str(i) for i in range(p)],
                dtype=np.float64,
            )
            precompute_var_covar_cholesky(
                self.covariates._data,
                self.haplotype_variables._data,
                L_out=self.L_precomp._data,
                diag_out=self.diag_precomp._data,
                batch_size=self.batch_size
            )
        else:
            self._precomputed_factors = False
            self.L_precomp = None
            self.diag_precomp = None

        self.results = None
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=exists_ok)

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
        colnames = ['lod_score', 'lrt_pval',
            *(f'{var_name}_lod_score' for var_name in variable_names),
            *(f'{var_name}_pval' for var_name in variable_names),
            *(f'{var_name}_coef' for var_name in variable_names),
        ]
        return colnames

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
            gene_id=gene_id,
            covariates=self.covariates,
            bin_size=self.bin_size,
            L_precomp=self.L_precomp,
            diag_precomp=self.diag_precomp,
        )
        return e.test_all_positions(batch_size=self.batch_size)

    @staticmethod
    def apply_fdr_correction(results, inplace=True):
        raise ValueError('not implemented')

    def run_all_genes(self, processes=1, torch_threads_per_process=1, resume=False):

        def _run_gene_parallel(gene_id):

            os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads_per_process)
            os.environ["OMP_NUM_THREADS"] = str(torch_threads_per_process)
            os.environ["MKL_NUM_THREADS"] = str(torch_threads_per_process)
            import torch
            torch.set_num_threads(torch_threads_per_process)
            
            expression_variables = self._mm.load('expression_variables')
            covariates = self._mm.load('covariates')
            haplotype_variables = self._mm.load('haplotype_variables')
            if self._precomputed_factors:
                L_precomp = self._mm.load('L_precomp')
                diag_precomp = self._mm.load('diag_precomp')
            else:
                L_precomp, diag_precomp = None, None

            e = SinglePhenoQTLAnalysis(
                y=expression_variables.loc[gene_id],
                haplotype_variables=haplotype_variables,
                gene_id=gene_id,
                covariates=covariates,
                bonf_correction=len(expression_variables), # control for n genes
                bin_size=self.bin_size,
                minP_threshold=self.minP_threshold,
                L_precomp=L_precomp,
                diag_precomp=diag_precomp,
            )
            gene_results = e.test_all_positions(batch_size=self.batch_size)
            gene_results.write_parquet(os.path.join(self.output_dir, f'{gene_id}.parquet'))

        if resume:
            prev_results = eQTLResults(self.output_dir)
            gene_ids_to_test = [g_id for g_id in self.gene_ids if g_id not in prev_results]
        else:
            gene_ids_to_test = self.gene_ids

        gene_progress = progress_bar(
            gene_ids_to_test,
            label='Running eQTL',
            item_show_func=str,
        )
        with gene_progress:
            joblib.Parallel(n_jobs=processes, backend='loky')(
                joblib.delayed(_run_gene_parallel)(gene_id)
                for gene_id in gene_progress
            )

        self.results = eQTLResults(self.output_dir)
        return self.results
