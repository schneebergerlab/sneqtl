from logging import getLogger
import numpy as np
import pandas as pd

from snco.records import PredictionRecords
from snco.defaults import DEFAULT_RANDOM_SEED

from . import utils
from .eqtl import SNeQTLAnalysis
from ..io.misc import read_cb_stats
from ..io.gtf import read_gtf_gene_locs
from ..io.mtx import read_expression_matrix
from ..covars import get_principal_components, create_celltype_clusters


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
log = getLogger('sneqtl')


def run_eqtl(exprs_mat_dir, haplo_json_fn, cb_stats_fn, output_prefix, gtf_fn=None,
             min_cells_exprs=0.02,
             control_principal_components=True, min_pc_var_explained=0.01,
             max_pc_haplotype_var_explained=0.05, covar_names=None,
             model_parental_genotype=False,
             celltype_haplotype_interaction=False, celltype_n_clusters=None,
             control_haplotypes=None, control_cis_haplotype=False,
             control_haplotypes_r2=0.95, cb_filter_exprs=None,
             run_peakcall=True, lod_threshold=5, rel_lod_threshold=0.1,
             pval_threshold=1e-2, rel_prominence=0.25, ci_lod_drop=1.5,
             min_dist_between_eqtls=3e6, cis_eqtl_range=5e5,
             processes=1, rng=DEFAULT_RNG):

    if cb_stats_fn is not None:
        cb_stats = read_cb_stats(cb_stats_fn, cb_filter_exprs)
        cb_whitelist = cb_stats.index.tolist()
        log.info(f'Read {len(cb_whitelist)} barcodes')
    else:
        cb_stats = None
        cb_whitelist = None
    exprs_mat = read_expression_matrix(
        exprs_mat_dir, cb_whitelist,
        rel_min_cells_exprs=min_cells_exprs
    )
    log.info(f'Identified {len(exprs_mat)} genes to be tested')

    haplotypes = PredictionRecords.read_json(haplo_json_fn)
    log.info(f'Read haplotype information in {haplotypes.bin_size/1e3:.0f} kb bins')
    log.info(f'There are {len(haplotypes.nbins)} chromosomes and '
             f'{sum(haplotypes.nbins.values())} bins to be tested')
    haplotypes = haplotypes.to_frame(dtype=np.float16).loc[cb_whitelist]

    principal_components = get_principal_components(
        exprs_mat, haplotypes,
        min_var_explained=min_pc_var_explained,
        max_pc_haplotype_var_explained=max_pc_haplotype_var_explained,
        rng=rng,
    )

    if covar_names is not None and cb_stats is not None:
        covariates = pd.concat([
            principal_components,
            utils.convert_to_numeric(cb_stats.loc[:, covar_names], drop_first=True)
        ], axis=1)
    else:
        covariates = principal_components

    if celltype_haplotype_interaction:
        celltype_clusters = create_celltype_clusters(
            principal_components,
            n_clusters=celltype_n_clusters,
            rng=rng
        )
    else:
        celltype_clusters = None

    if model_parental_genotype and cb_stats is not None:
        parental_genotypes = cb_stats['geno_pred']
        n_geno = len(parental_genotypes.unique())
        if n_geno == 1:
            log.warn('All parental (diploid) genotypes are the same. Genotype will not be modelled')
            parental_genotypes = None
        else:
            log.info(f'There are {n_geno} parental (diploid) genotypes to be tested')

    else:
        log.info('No parental (diploid) genotypes provided. Genotypes are assumed to be the same for all nuclei')
        parental_genotypes = None

    if control_cis_haplotype and gtf_fn is not None:
        gene_locs = read_gtf_gene_locs(gtf_fn)
        log.info(f'Read gene locations from GTF file {gtf_fn}')
    else:
        if control_cis_haplotype:
            log.warn('No GTF file provided, cannot control for cis haplotypes!')
        gene_locs = None

    e = SNeQTLAnalysis(
        exprs_mat, haplotypes, cb_whitelist,
        covariates=covariates,
        interacting_variables=celltype_clusters,
        parental_genotypes=parental_genotypes,
        control_haplotypes=control_haplotypes,
        control_cis_haplotype=control_cis_haplotype,
        gene_locs=gene_locs,
        control_haplotypes_r2=control_haplotypes_r2,
    )
    example_gene_id = e.gene_ids[0]
    example_covars = e.get_covariates(example_gene_id).columns.tolist()
    example_covars = [c.replace(example_gene_id, 'gene_id') for c in example_covars]
    haplo_test_vars = e.param_names
    log.debug(f'Controlling for {len(example_covars)} covariates: {example_covars}')
    log.debug(f'{len(haplo_test_vars)} haplotype-associated variables will be tested: {haplo_test_vars}')
    log.debug(f'Estimated {e.effective_haplotypes} effective haplotypes - this number will be used '
              'for haplotype-wise multiple testing correction')

    eqtl_results = e.run_all_genes(processes=processes)
    
    log.info(f'eQTL analysis complete - writing full results to {output_prefix}.eqtls.tsv')
    eqtl_results.to_csv(
        f'{output_prefix}.eqtls.tsv',
        sep='\t',
        index=False,
        header=True,
        float_format='%.4g'
    )
    if run_peakcall:
        if gtf_fn is None:
            log.warn('GTF file was not provided, cannot run peakcall')
        else:
            from ..peaks import  run_peakcall
            run_peakcall(
                eqtl_tsv_fn=None,
                output_tsv_fn=f'{output_prefix}.eqtl_peaks.tsv',
                gtf_fn=gtf_fn,
                eqtl_results=eqtl_results,
                peak_variable='overall',
                lod_threshold=lod_threshold,
                rel_lod_threshold=rel_lod_threshold,
                pval_threshold=pval_threshold,
                rel_prominence=rel_prominence,
                ci_lod_drop=ci_lod_drop,
                min_dist_between_eqtls=min_dist_between_eqtls,
                cis_eqtl_range=cis_eqtl_range
            )
