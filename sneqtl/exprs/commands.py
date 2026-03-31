from logging import getLogger
import numpy as np
import pandas as pd

from coelsch.records import PredictionRecords
from coelsch.defaults import DEFAULT_RANDOM_SEED

from sneqtl.io.misc import read_cb_stats
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.variables.pca import PrincipalComponents
from sneqtl.variables.celltypes import CellTypes
from sneqtl.variables.var_utils import concat
from .load import parallel_scrna_gene_counter
from .doublet import predict_expression_doublets

log = getLogger('sneqtl')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def run_loadexprs(output_mat_dir, bam_fn, gtf_fn, cb_stats_fn=None,
                  min_mapq=255, cb_tag='CB', umi_tag='UB', umi_collapse_method='exact',
                  strand_type='fwd', quant_type='gene', cb_filter_exprs=None, processes=1):
    if cb_stats_fn is not None:
        cb_stats = read_cb_stats(cb_stats_fn, cb_filter_exprs)
        cb_whitelist = cb_stats.index.tolist()
        log.info(f'Read {len(cb_whitelist)} barcodes from "{cb_stats_fn}"')
    exprs = parallel_scrna_gene_counter(
        bam_fn, gtf_fn, cb_whitelist,
        min_mapq=min_mapq,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        umi_dedup_method=umi_collapse_method,
        strand_filt_type=strand_type,
        quant_type=quant_type,
        njobs=processes
    )
    exprs.write_matrix_market(output_mat_dir)


def run_concatexprs(output_mat_dir, exprs_mat_dir, merge_suffixes=None):
    '''
    Concatenates expression matrices, potentially from different datasets, 
    adding suffixes to cell barcodes to avoid name collisions
    '''
    exprs_objects = []
    for path in exprs_mat_dir:
        e = ExpressionVariables.read_matrix_market(path)
        exprs_objects.append(e)
    concat_mtx = concat(*exprs_objects, suffixes=merge_suffixes, axis=1)
    concat_mtx.write_matrix_market(output_mat_dir)


def run_exprsstats(exprs_mat_dir, cb_stats_fn, output_tsv_fn,
                   n_doublets=0.25, k_neighbours=100,
                   rel_min_cells_exprs=0.02, abs_min_cells_exprs=100,
                   min_pc_var_explained=0.01,
                   celltype_n_clusters=None, cb_filter_exprs=None):
    if cb_stats_fn is not None:
        cb_stats = read_cb_stats(cb_stats_fn, cb_filter_exprs)
        cb_whitelist = list(cb_stats.index)
        log.info(f'Read {len(cb_whitelist)} barcodes from {cb_stats_fn}')
    else:
        cb_stats = None
        cb_whitelist = None
    exprs_mat = ExpressionVariables.read_matrix_market(exprs_mat_dir)
    if cb_whitelist is not None:
        cb_whitelist = sorted(set(exprs_mat.barcodes).intersection(cb_whitelist))
        if not cb_whitelist:
            raise ValueError('No overlap between exprs_mat barcodes and cb_whitelist barcodes')
        exprs_mat = exprs_mat.reindex(columns=cb_whitelist)
        cb_stats = cb_stats.reindex(index=cb_whitelist)
    exprs_mat = exprs_mat.filter_genes(
        rel_min_cells_exprs=rel_min_cells_exprs,
        abs_min_cells_exprs=abs_min_cells_exprs,
    ).log_normalise()
    log.info(f'Read expression matrix with {exprs_mat.shape[0]} genes and {exprs_mat.shape[1]} barcodes')

    principal_components = PrincipalComponents.from_expression(
        exprs_mat,
        min_var_explained=min_pc_var_explained,
        rng=DEFAULT_RNG,
    )

    celltype_clusters = CellTypes.from_principal_components(
        principal_components,
        n_clusters=celltype_n_clusters,
        rng=DEFAULT_RNG
    )

    doublet_pred = predict_expression_doublets(
        exprs_mat, principal_components, celltype_clusters,
        n_doublets=n_doublets,
        k_neighbours=k_neighbours,
    )

    # convert to dataframes for output
    exprs_cb_stats = principal_components.to_dataframe()
    exprs_cb_stats['celltype_cluster'] = celltype_clusters.to_dataframe().idxmax(axis=1)
    exprs_cb_stats['expression_doublet_probability'] = doublet_pred

    if cb_stats is not None:
        cb_stats = pd.concat([cb_stats, exprs_cb_stats], axis=1)
    else:
        cb_stats = exprs_cb_stats
    log.info(f'Writing new stats to {output_tsv_fn}')
    cb_stats.to_csv(output_tsv_fn, sep='\t')
    return cb_stats