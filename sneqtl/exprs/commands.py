from logging import getLogger

from snco.records import PredictionRecords

from sneqtl.io.misc import read_cb_stats
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.variables.var_utils import concat
from .load import parallel_scrna_gene_counter
from .doublet import predict_expression_doublets

log = getLogger('sneqtl')


def run_loadexprs(output_mat_dir, bam_fn, gtf_fn, cb_whitelist_fn,
                  min_mapq=255, cb_tag='CB', umi_tag='UB', umi_collapse_method='exact',
                  strand_type='fwd', quant_type='gene', processes=1):
    exprs = parallel_scrna_gene_counter(
        bam_fn, gtf_fn, cb_whitelist_fn,
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
    concat_mtx = concat(mtx_objects, merge_suffixes, axis=1)
    concat_mtx.write_matrix_market(output_mat_dir)


def run_doubletexprs(exprs_mat_dir, cb_stats_fn, output_tsv_fn,
                     n_doublets=0.25, k_neighbours=100, min_cells_exprs=0.02,
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
        exprs_mat = exprs_mat.reindex(columns=cb_whitelist)
    exprs_mat = exprs_mat.filter_genes().log_normalise()
    log.info(f'Read expression matrix with {exprs_mat.shape[0]} genes and {exprs_mat.shape[1]} barcodes')

    doublet_pred = predict_expression_doublets(
        exprs_mat,
        n_doublets=n_doublets,
        k_neighbours=k_neighbours,
        min_var_explained=min_pc_var_explained,
        n_clusters=celltype_n_clusters
    )
    if cb_stats is not None:
        cb_stats['expression_doublet_probability'] = doublet_pred
    else:
        cb_stats = doublet_pred.to_frame()
    log.info(f'Writing new stats to {output_tsv_fn}')
    cb_stats.to_csv(output_tsv_fn, sep='\t')
    return cb_stats