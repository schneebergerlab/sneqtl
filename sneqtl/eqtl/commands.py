from logging import getLogger
import numpy as np

from coelsch.records import PredictionRecords
from coelsch.experiment.genotypes import GenotypeKey
from coelsch.defaults import DEFAULT_RANDOM_SEED

from .memmap import MemmapManager
from .cache import AnalysisCache
from .preprocessing import preprocess_variables, eQTLVariables
from .analysis import eQTLAnalysisManager


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
log = getLogger('sneqtl')


def run_eqtl(exprs_mat_dir, haplo_json_fn, cb_stats_fn, output_prefix,
             diploid_json_fn=None, gtf_fn=None, vcf_fn=None,
             rel_min_cells_exprs=0.02, abs_min_cells_exprs=100,
             control_principal_components=True, min_pc_var_explained=0.01, covar_names=None,
             genotype_covariate_mode='none', reference_genotype=None,
             min_cells_per_geno=100, variant_maf=0.05, genotype_covariates_as_pcs=False,
             celltype_haplotype_interaction=False, celltype_n_clusters=None,
             control_cis_haplotypes=True, cis_hap_window_size=250_000, covariate_haplotypes=None,
             cb_filter_exprs=None, minP_threshold=0.01, n_perm=1000, 
             run_peakcall=True, lod_threshold=5, rel_lod_threshold=0.1,
             rel_prominence=0.25, ci_lod_drop=1.5,
             min_dist_between_eqtls=3e6, cis_eqtl_range=5e5,
             processes=1, torch_threads_per_process=1,
             resume=False, rng=DEFAULT_RNG,
             cache_dir=None, keep_variables=False):

    if keep_variables and cache_dir is None:
        raise ValueError('keep_variables=True requires cache_dir')

    # collect all parameters for cache checking
    params = locals().copy()
    params['rng_state'] = rng.bit_generator.state
    params['precompute_hap_factors'] = True

    cache = AnalysisCache(cache_dir, parameters=params)

    with cache.lock():
        if keep_variables and not cache.is_complete():
            cache.reset()

        mm = MemmapManager(cache, persist=keep_variables)

        if mm.cached:
            log.info(f'Reusing cached variables: {cache.path}')
            variables = eQTLVariables(
                mm=mm,
                expression_variables=mm.load('expression_variables'),
                covariates=mm.load('covariates'),
                haplotype_variables=mm.load('haplotype_variables'),
                cis_haplo_covariates=(
                    mm.load('cis_haplo_covariates')
                    if 'cis_haplo_covariates' in cache.datasets else None
                ),
                L_precomp=(
                    mm.load('L_precomp')
                    if 'L_precomp' in cache.datasets else None
                ),
                diag_precomp=(
                    mm.load('diag_precomp')
                    if 'diag_precomp' in cache.datasets else None
                ),
            )
        else:
            try:
                variables = preprocess_variables(
                    exprs_mat_dir, haplo_json_fn, cb_stats_fn, mm,
                    diploid_json_fn=diploid_json_fn,
                    gtf_fn=gtf_fn,
                    vcf_fn=vcf_fn,
                    rel_min_cells_exprs=rel_min_cells_exprs,
                    abs_min_cells_exprs=abs_min_cells_exprs,
                    control_principal_components=control_principal_components,
                    min_pc_var_explained=min_pc_var_explained,
                    covar_names=covar_names,
                    genotype_covariate_mode=genotype_covariate_mode,
                    reference_genotype=reference_genotype,
                    min_cells_per_geno=min_cells_per_geno,
                    variant_maf=variant_maf,
                    genotype_covariates_as_pcs=genotype_covariates_as_pcs,
                    celltype_haplotype_interaction=celltype_haplotype_interaction,
                    celltype_n_clusters=celltype_n_clusters,
                    control_cis_haplotypes=control_cis_haplotypes,
                    cis_hap_window_size=cis_hap_window_size,
                    covariate_haplotypes=covariate_haplotypes,
                    cb_filter_exprs=cb_filter_exprs,
                    precompute_hap_factors=True,
                    rng=rng                
                )

                if keep_variables:
                    cache.mark_complete()

            except Exception:
                mm.remove_all()
                if keep_variables:
                    cache.reset()
                raise


    output_dir = f'{output_prefix}.eqtls'
    with eQTLAnalysisManager(
        output_dir=output_dir,
        variables=variables,
        minP_threshold=minP_threshold,
        n_perm=n_perm,
        exists_ok=resume,
    ) as e:
        covar_names = list(e.covariates.variable_names)
        if e.control_cis_haplotypes:
            covar_names += e.cis_haplo_covariates.variable_names
        haplo_var_names = e.haplotype_variable_names
        log.debug(f'Controlling for {len(covar_names)} covariates: {covar_names}')
        log.debug(f'{len(haplo_var_names)} haplotype-associated variables will be tested: {haplo_var_names}')

        eqtl_results = e.run_all_genes(
            processes=processes,
            torch_threads_per_process=torch_threads_per_process,
            resume=resume,
        )

    if run_peakcall:
        if gtf_fn is None:
            log.warn('GTF file was not provided, cannot run peakcall')
        else:
            from ..peaks import run_peakcall
            run_peakcall(
                eqtl_dir=None,
                output_tsv_fn=f'{output_prefix}.eqtl_peaks.tsv',
                gtf_fn=gtf_fn,
                eqtl_results=eqtl_results,
                peak_variable='overall',
                lod_threshold=lod_threshold,
                rel_lod_threshold=rel_lod_threshold,
                rel_prominence=rel_prominence,
                ci_lod_drop=ci_lod_drop,
                min_dist_between_eqtls=min_dist_between_eqtls,
                cis_eqtl_range=cis_eqtl_range
            )
