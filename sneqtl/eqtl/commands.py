from logging import getLogger
import numpy as np

from snco.defaults import DEFAULT_RANDOM_SEED

from sneqtl.variables.base import VariableManager
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.variables.haplotypes import HaplotypeVariables
from sneqtl.variables.genotypes import ParentalGenotypesContainer
from sneqtl.variables.pca import PrincipalComponents
from sneqtl.variables.celltypes import CellTypes
from sneqtl.variables.var_utils import concat
from sneqtl.io.misc import read_cb_stats
from .analysis import eQTLAnalysisManager


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
log = getLogger('sneqtl')


def run_eqtl(exprs_mat_dir, haplo_json_fn, cb_stats_fn, output_prefix, gtf_fn=None,
             rel_min_cells_exprs=0.02, abs_min_cells_exprs=100,
             control_principal_components=True, min_pc_var_explained=0.01,
             covar_names=None, model_parental_genotype=False, reference_genotype=None,
             celltype_haplotype_interaction=False, celltype_n_clusters=None,
             covariate_haplotypes=None, cb_filter_exprs=None,
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
    expression_variables = ExpressionVariables.read_matrix_market(exprs_mat_dir)
    if cb_whitelist is not None:
        expression_variables = expression_variables.reindex(columns=cb_whitelist)
    expression_variables = (
        expression_variables.filter_genes(
            rel_min_cells_exprs=rel_min_cells_exprs,
            abs_min_cells_exprs=abs_min_cells_exprs,
        ).log_normalise()
    )
    log.info(f'Identified {len(expression_variables)} genes to be tested')

    haplotypes = HaplotypeVariables.read_prediction_records(haplo_json_fn, cb_whitelist)
    log.info(f'Read haplotype information in {haplotypes.bin_size/1e3:.0f} kb bins')
    log.info(f'There are {len(haplotypes.chroms)} chromosomes and '
             f'{len(haplotypes.columns)} bins to be tested')

    if expression_variables.barcodes != haplotypes.barcodes:
        cb_whitelist = sorted(
            set(expression_variables.barcodes).intersection(haplotypes.barcodes).intersection(cb_whitelist)
        )
        expression_variables = expression_variables.reindex(columns=cb_whitelist)
        haplotypes = haplotypes.reindex(index=cb_whitelist)
        cb_stats = cb_stats.reindex(index=cb_whitelist)
        assert expression_variables.barcodes == haplotypes.barcodes
    else:
        cb_whitelist = list(expression_variables.barcodes)
    principal_components = PrincipalComponents.from_expression(
        expression_variables,
        min_var_explained=min_pc_var_explained,
        rng=rng,
    )
    log.info(f'Identified {principal_components.shape[1]} principal components '
             f'explaining {principal_components.var_explained * 100:.1f}% of variance')
    principal_components = principal_components.filter_correlated(haplotypes, max_r2=0.05)
    if control_principal_components:
        log.info('After filtering for PCs correlated with haplotypes, '
                 f'{principal_components.columns} will be used as covariates')
    if celltype_haplotype_interaction:
        celltype_clusters = CellTypes.from_principal_components(
            principal_components,
            n_clusters=celltype_n_clusters,
            rng=rng
        )
        log.info(f'Identified {celltype_clusters.shape[1]} cell type clusters')
        log.debug(celltype_clusters.celltype_counts)
    else:
        celltype_clusters = None

    if control_principal_components:
        covariates = principal_components
    else:
        covariates = None

    if covar_names is not None and cb_stats is not None:
        extra_covars = (VariableManager.from_dataframe(cb_stats.loc[cb_whitelist, covar_names])
                                       .as_numeric(drop_first=True))
        if covariates is None:
            covariates = extra_covars
        else:
            covariates = concat(principal_components, extra_covars, axis=1)

    if model_parental_genotype and cb_stats is not None:
        parental_genotypes = cb_stats['geno_pred'].to_dict()
        # TODO: fails for nans...
        n_geno = len(set(parental_genotypes.values()))
        if n_geno == 1:
            log.warn('All parental (diploid) genotypes are the same. Genotype will not be modelled')
            parental_genotypes = None
        else:
            log.info(f'There are {n_geno} parental (diploid) genotypes to be tested')
            parental_genotypes = ParentalGenotypesContainer.from_genotype_dict(
                parental_genotypes, reference_genotype=reference_genotype)

    else:
        log.info('No parental (diploid) genotypes provided. Genotypes are assumed to be the same for all nuclei')
        parental_genotypes = None

    interacting_variables = [celltype_clusters,] if celltype_clusters is not None else None

    e = eQTLAnalysisManager(
        expression_variables=expression_variables,
        haplotypes=haplotypes,
        parental_genotypes=parental_genotypes,
        interacting_variables=interacting_variables,
        covariates=covariates,
        covariate_haplotypes=covariate_haplotypes,
    )
    covar_names = e.covariates.variable_names
    haplo_var_names = e.haplotype_variable_names
    log.debug(f'Controlling for {len(covar_names)} covariates: {covar_names}')
    log.debug(f'{len(haplo_var_names)} haplotype-associated variables will be tested: {haplo_var_names}')

    eqtl_results = e.run_all_genes(processes=processes).to_dataframe()
    
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
