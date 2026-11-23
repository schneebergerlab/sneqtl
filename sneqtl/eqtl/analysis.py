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
from pyarrow import compute as pc

from coelsch.main.logger import progress_bar
from .model.qtl import SinglePhenoQTLAnalysis, SingleGeneResults
from .preprocessing import eQTLVariables


class eQTLResults:
    def __init__(self, results_dir):
        self._dir = results_dir
        self.gene_ids = {
            os.path.splitext(e.name)[0]: e.path
            for e in os.scandir(results_dir)
            if e.is_file() and e.name.endswith(".parquet")
        }

        try:
            any_file = next(iter(self.gene_ids.values()))
            self.schema = pq.read_schema(any_file)
            self.bin_size = int(
                self.schema.field("pos").metadata[b"bin_size"].decode()
            )
        except StopIteration:
            self.schema = None
            self.bin_size = None

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

    def get_haplotype_results(self, chrom, pos):
        first_file = next(iter(self.gene_ids.values()))
        pf = pq.ParquetFile(first_file)

        # Find the row group and row within it.
        for rg in range(pf.num_row_groups):
            coords = pf.read_row_group(rg, columns=["chrom", "pos"])

            mask = pc.and_(
                pc.equal(coords["chrom"], chrom),
                pc.equal(coords["pos"], pos)
            )
            idx = pc.indices_nonzero(mask)

            if len(idx):
                if len(idx) > 1:
                    raise ValueError(
                        f"Multiple rows found for chrom={chrom}, pos={pos}"
                    )
                row_idx = idx[0].as_py()
                break
        else:
            raise KeyError(f"No row found for chrom={chrom}, pos={pos}")

        return pd.concat(
            [
                pq.ParquetFile(path)
                .read_row_group(rg)
                .slice(row_idx, 1)
                .to_pandas()
                for path in self.gene_ids.values()
            ],
            ignore_index=True,
        )


class eQTLAnalysisManager:

    def __init__(self, output_dir, variables: eQTLVariables,
                 minP_threshold=0.01, n_perm=1000,
                 batch_size=500, exists_ok=False):

        self._mm = variables.mm

        self.expression_variables = variables.expression_variables
        self.covariates = variables.covariates
        self.cis_haplo_covariates = variables.cis_haplo_covariates
        self.haplotype_variables = variables.haplotype_variables
        self.L_precomp = variables.L_precomp
        self.diag_precomp = variables.diag_precomp
        self.bin_size = variables.haplotype_variables.bin_size

        self.control_cis_haplotypes = self.cis_haplo_covariates is not None
        self._precomputed_factors = self.L_precomp is not None

        self.batch_size = batch_size
        self.minP_threshold = minP_threshold
        self.n_perm = n_perm

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

    def run_gene(self, gene_id):
        if self.control_cis_haplotypes:
            cis_haplo_covariates = self.cis_haplo_covariates.get_gene(gene_id)
        else:
            cis_haplo_covariates = None

        e = SinglePhenoQTLAnalysis(
            y=self.expression_variables.loc[gene_id],
            haplotype_variables=self.haplotype_variables,
            gene_id=gene_id,
            covariates=self.covariates,
            cis_haplo_covariates=cis_haplo_covariates,
            bonf_correction=len(self.expression_variables),
            bin_size=self.bin_size,
            minP_threshold=self.minP_threshold,
            n_perm=self.n_perm,
            L_precomp=self.L_precomp,
            diag_precomp=self.diag_precomp,
        )
        return e.test_all_positions(batch_size=self.batch_size)

    def run_all_genes(self, processes=1, torch_threads_per_process=1, resume=False):

        def _run_gene_parallel(gene_id):

            os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads_per_process)
            os.environ["OMP_NUM_THREADS"] = str(torch_threads_per_process)
            os.environ["MKL_NUM_THREADS"] = str(torch_threads_per_process)
            import torch
            torch.set_num_threads(torch_threads_per_process)
            
            expression_variables = self._mm.load('expression_variables')
            covariates = self._mm.load('covariates')
            if self.control_cis_haplotypes:
                cis_haplo_covariates = self._mm.load('cis_haplo_covariates')
                cis_haplo_covariates = cis_haplo_covariates.get_gene(gene_id)
            else:
                cis_haplo_covariates = None
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
                cis_haplo_covariates=cis_haplo_covariates,
                bonf_correction=len(expression_variables), # control for n genes
                bin_size=self.bin_size,
                n_perm=self.n_perm,
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
