from collections import Counter
from dataclasses import dataclass
from logging import getLogger

import numpy as np

from coelsch.defaults import DEFAULT_RANDOM_SEED
from coelsch.records import PredictionRecords

from sneqtl.io.gtf import read_gtf_gene_locs
from sneqtl.io.misc import read_cb_stats
from sneqtl.variables.base import VariableManager
from sneqtl.variables.celltypes import CellTypes
from sneqtl.variables.covariates import Covariates, CisHaploCovariates
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.variables.genotypes import ParentalGenotypesContainer
from sneqtl.variables.haplotypes import HaplotypeVariables
from sneqtl.variables.interactions import HaplotypeInteractionVariables
from sneqtl.variables.pca import PrincipalComponents
from sneqtl.variables.variants import VariantVariables
from sneqtl.variables.var_utils import check_aligned, concat

from .memmap import MemmapManager
from .model.lm import precompute_var_covar_cholesky


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
log = getLogger('sneqtl')


@dataclass
class eQTLVariables:
    mm: MemmapManager
    expression_variables: ExpressionVariables
    covariates: Covariates
    haplotype_variables: HaplotypeInteractionVariables
    cis_haplo_covariates: CisHaploCovariates | None
    L_precomp: VariableManager | None
    diag_precomp: VariableManager | None

    def __post_init__(self):
        check_aligned(
            self.expression_variables.T,
            self.covariates,
            self.haplotype_variables.T,
            self.cis_haplo_covariates,
        )

        if self.cis_haplo_covariates is not None:
            if self.expression_variables.gene_ids != self.cis_haplo_covariates.gene_ids:
                raise ValueError('Gene IDs are not aligned')


def read_genotype_variables(haplo_json_fn, diploid_json_fn=None,
                            reference_genotype=None, genotype_covariate_mode='none',
                            cb_whitelist=None, min_cells_per_geno=100):

    haploid_records = PredictionRecords.read_json(haplo_json_fn, subset=cb_whitelist)
    bin_size = haploid_records.bin_size

    if 'genotypes' not in haploid_records.metadata:
        raise ValueError('PredictionRecords object must contain a "genotypes" metadata slot')

    def _strip_geno_name(geno_str):
        return geno_str.split(':', maxsplit=1)[-1]

    parental_genotypes = {
        cb: _strip_geno_name(haploid_records.metadata['genotypes'][cb])
        for cb in haploid_records.barcodes
    }
    keep_genotypes = {
        geno for geno, count in Counter(parental_genotypes.values()).items()
        if count >= min_cells_per_geno
    }

    parental_genotypes = {cb: geno for cb, geno in parental_genotypes.items() if geno in keep_genotypes}
    cb_whitelist = list(parental_genotypes)

    if diploid_json_fn is not None:
        diploid_records = PredictionRecords.read_json(diploid_json_fn)
    else:
        diploid_records = None

    haplotypes = HaplotypeVariables.from_prediction_records(haploid_records, diploid_records, cb_whitelist)

    
    log.info(f'Read {len(cb_whitelist)} barcodes')
    log.info(f'Read haplotype information in {bin_size/1e3:.0f} kb bins')
    log.info(f'There are {len(haplotypes.chroms)} chromosomes and '
             f'{len(haplotypes.columns)} bins to be tested')

    n_geno = len(keep_genotypes)
    if genotype_covariate_mode != 'none' and n_geno == 1:
        log.warn('All parental (diploid) genotypes are the same. Genotype will not be modelled')
        genotype_covariate_mode = 'none'
    else:
        log.info(f'There are {n_geno} parental (diploid) genotypes to be tested')

    parental_genotypes = ParentalGenotypesContainer.from_genotype_dict(
        parental_genotypes, reference_genotype=reference_genotype,
        covariate_mode=genotype_covariate_mode
    )

    return haplotypes, parental_genotypes, bin_size, cb_whitelist


def read_expression_variables(exprs_mat_dir, cb_whitelist=None,
                              rel_min_cells_exprs=0.02, abs_min_cells_exprs=100):

    expression_variables = ExpressionVariables.read_matrix_market(exprs_mat_dir)

    if cb_whitelist is not None:
        cb_whitelist = sorted(
            set(expression_variables.barcodes).intersection(cb_whitelist)
        )
        expression_variables = expression_variables.reindex(columns=cb_whitelist)

    expression_variables = (
        expression_variables.filter_genes(
            rel_min_cells_exprs=rel_min_cells_exprs,
            abs_min_cells_exprs=abs_min_cells_exprs,
        ).log_normalise()
    )
    log.info(f'Identified {len(expression_variables)} genes to be tested')

    return expression_variables, cb_whitelist


def read_variant_variables(vcf_fn, haplotypes, parental_genotypes, reference_genotype, variant_maf=0.05):
    if reference_genotype is None:
        raise ValueError('If a VCF file is provided, the reference genotype name must also be provided')
    variants = VariantVariables.read_vcf(
        vcf_fn, bin_size=haplotypes.bin_size,
        reference_name=reference_genotype,
        use_haplotypes=parental_genotypes.columns
    )
    variants = variants.filter_minor_allele(
        parental_genotypes,
        haplotypes,
        min_frac=variant_maf
    )
    log.info(f'Identified {len(variants)} variant clusters above {variant_maf * 100:.1f}% minor allele '
             'fraction - these will be modelled instead of haplotype')
    return variants


def calculate_principal_components(expression_variables, cb_stats=None,
                                    min_var_explained=0.01, rng=DEFAULT_RNG):

    if cb_stats is not None and cb_stats.columns.str.startswith('pc_').any():
        pc = cb_stats.loc[:, cb_stats.columns.str.startswith('pc_')]
        pc.columns = pc.columns.str[3:]
        log.info(f'Using {len(pc.columns)} predetermined principal components from stats file')
        principal_components = PrincipalComponents.from_dataframe(pc)
        return principal_components


    principal_components = PrincipalComponents.from_expression(
        expression_variables,
        min_var_explained=min_var_explained,
        rng=rng,
    )
    log.info(f'Identified {principal_components.shape[1]} principal components '
             f'explaining {principal_components.var_explained * 100:.1f}% of variance')

    return principal_components


def calculate_celltype_clusters(expression_variables, principal_components=None,
                                cb_stats=None, celltype_n_clusters=None,
                                min_var_explained=0.01, rng=DEFAULT_RNG):

    if cb_stats is not None and cb_stats.columns.str.startswith('ct_').any():
        ct = cb_stats.loc[:, cb_stats.columns.str.startswith('ct_')]
        ct.columns = ct.columns.str[3:]
        log.info(f'Using {len(ct.columns)} predetermined celltypes from stats file')
        celltype_clusters = CellTypes.from_dataframe(ct)
        log.debug(celltype_clusters.celltype_counts)
        return celltype_clusters

    if principal_components is None:
        principal_components = calculate_principal_components(
            expression_variables, cb_stats, min_var_explained, rng
        )

    celltype_clusters = CellTypes.from_principal_components(
        principal_components,
        n_clusters=celltype_n_clusters,
        rng=rng
    )
    log.info(f'Identified {celltype_clusters.shape[1]} cell type clusters')
    log.debug(celltype_clusters.celltype_counts)
    return celltype_clusters


def preprocess_variables(exprs_mat_dir, haplo_json_fn, cb_stats_fn, mm,
                         diploid_json_fn=None, gtf_fn=None, vcf_fn=None,
                         rel_min_cells_exprs=0.02, abs_min_cells_exprs=100,
                         control_principal_components=True, min_pc_var_explained=0.01,
                         covar_names=None, genotype_covariate_mode='none',
                         reference_genotype=None, min_cells_per_geno=100,
                         variant_maf=0.05, genotype_covariates_as_pcs=False,
                         celltype_haplotype_interaction=False, celltype_n_clusters=None,
                         control_cis_haplotypes=True, cis_hap_window_size=250_000,
                         covariate_haplotypes=None, cb_filter_exprs=None,
                         precompute_hap_factors=True, batch_size=500, rng=DEFAULT_RNG):

    if cb_stats_fn is not None:
        cb_stats = read_cb_stats(cb_stats_fn, cb_filter_exprs)
        cb_whitelist = cb_stats.index.tolist()
    else:
        cb_stats = None
        cb_whitelist = None

    haplotypes, parental_genotypes, bin_size, cb_whitelist = read_genotype_variables(
        haplo_json_fn, diploid_json_fn, reference_genotype,
        cb_whitelist=cb_whitelist,
        genotype_covariate_mode=genotype_covariate_mode,
        min_cells_per_geno=min_cells_per_geno,
    )

    expression_variables, cb_whitelist = read_expression_variables(
        exprs_mat_dir,
        cb_whitelist=cb_whitelist,
        rel_min_cells_exprs=rel_min_cells_exprs,
        abs_min_cells_exprs=abs_min_cells_exprs
    )

    if expression_variables.barcodes != haplotypes.barcodes:
        cb_whitelist = sorted(
            set(expression_variables.barcodes).intersection(haplotypes.barcodes).intersection(cb_whitelist)
        )
        expression_variables = expression_variables.reindex(columns=cb_whitelist)
        haplotypes = haplotypes.reindex(index=cb_whitelist)
        parental_genotypes = parental_genotypes.reindex(index=cb_whitelist)

        if cb_stats is not None:
            cb_stats = cb_stats.reindex(index=cb_whitelist)
        assert expression_variables.barcodes == haplotypes.barcodes

    principal_components = None
    if control_principal_components:

        principal_components = calculate_principal_components(
            expression_variables, cb_stats,
            min_var_explained=min_pc_var_explained,
            rng=rng
        )
        principal_components = principal_components.filter_correlated(haplotypes, max_r2=0.05)
        log.info('After filtering for PCs correlated with haplotypes, '
                 f'{principal_components.columns} will be used as covariates')

    celltype_clusters = None
    if celltype_haplotype_interaction:
        celltype_clusters = calculate_celltype_clusters(
            expression_variables, principal_components, cb_stats,
            celltype_n_clusters=celltype_n_clusters,
            min_var_explained=min_pc_var_explained,
            rng=rng
        )

    extra_covars = None
    if covar_names is not None and cb_stats is not None:
        extra_covars = (VariableManager.from_dataframe(cb_stats.loc[cb_whitelist, covar_names])
                                       .as_numeric(drop_first=True))

    variants = None
    if vcf_fn is not None:
        variants = read_variant_variables(
            vcf_fn,
            haplotypes,
            parental_genotypes,
            reference_genotype,
            variant_maf=variant_maf
        )

    if covariate_haplotypes is not None and len(covariate_haplotypes):
        covariate_haplotypes, haplotypes = haplotypes.get_covariate_haplotypes(
            parental_genotypes, covariate_haplotypes, remove_correlated=True,
        )
    else:
        covariate_haplotypes = None

    covariates = principal_components
    if extra_covars is not None:
        if covariates is None:
            covariates = extra_covars
        else:
            covariates = concat(covariates, extra_covars, axis=1)

    interacting_variables = [celltype_clusters] if celltype_clusters is not None else []

    check_aligned(
        expression_variables.T, haplotypes, parental_genotypes,
        covariates, *interacting_variables,
    )

    # begin final assembly of datasets
    expression_variables.rename('expression_variables')
    expression_variables = mm.convert(expression_variables)

    # build covariates
    covariates = Covariates.from_variables(
        parental_genotypes=parental_genotypes,
        interacting_variables=interacting_variables,
        covariate_haplotypes=covariate_haplotypes,
        base_covariates=covariates,
        variants=variants,
        genotype_covariates_as_pcs=genotype_covariates_as_pcs,
        name='covariates',
        barcodes=haplotypes.barcodes,
    )

    HaplotypeInteractionVariables.validate_design(
        parental_genotypes=parental_genotypes,
        covariates=covariates,
        interacting_variables=interacting_variables,
        variant_mode=variants is not None,
    )

    covariates = mm.convert(covariates)

    cis_haplo_covariates = None
    if control_cis_haplotypes:
        if gtf_fn is None:
            raise ValueError('gtf_fn must be provided when controlling for cis haplotypes')
        gene_locs = read_gtf_gene_locs(gtf_fn)

        cis_haplo_covariates = CisHaploCovariates.from_haplotypes(
            haplotype_variables=haplotypes,
            gene_locs=gene_locs,
            parental_genotypes=parental_genotypes,
            variants=variants,
            window_size=cis_hap_window_size,
            gene_ids=expression_variables.gene_ids,
            name='cis_haplo_covariates'
        )
        cis_haplo_covariates = mm.convert(cis_haplo_covariates)

    haplotype_variables = HaplotypeInteractionVariables.from_cross_product(
        haplotype_variables=haplotypes,
        parental_genotypes=parental_genotypes,
        variant_variables=variants,
        interacting_variables=interacting_variables,
        name='haplotype_variables',
        memmap_manager=mm,
    )
    haplotype_variables._data.flush()

    # if we don't have gene-specific covariates, we can precompute some shared factors
    L_precomp = None
    diag_precomp = None
    if precompute_hap_factors and not control_cis_haplotypes:
        p = len(covariates.variable_names) + len(haplotype_variables.variable_names)
        L_precomp = mm.create_empty(
            'L_precomp',
            index=haplotype_variables.haplotypes,
            columns=[str(i) for i in range(p)],
            items=[str(i) for i in range(p)],
            dtype=np.float64,
        )
        diag_precomp = mm.create_empty(
            'diag_precomp',
            index=haplotype_variables.haplotypes,
            columns=[str(i) for i in range(p)],
            dtype=np.float64,
        )
        precompute_var_covar_cholesky(
            covariates._data,
            haplotype_variables._data,
            L_out=L_precomp._data,
            diag_out=diag_precomp._data,
            batch_size=batch_size
        )
        L_precomp._data.flush()
        diag_precomp._data.flush()

    return eQTLVariables(
        mm=mm,
        expression_variables=expression_variables,
        covariates=covariates,
        haplotype_variables=haplotype_variables,
        cis_haplo_covariates=cis_haplo_covariates,
        L_precomp=L_precomp,
        diag_precomp=diag_precomp,
    )