from logging import getLogger

from snco.records import PredictionRecords

from ..io.mtx import (
    read_matrix_market_as_pd_matrix,
    read_expression_matrix,
    write_matrix_market
)
from ..io.misc import read_cb_stats
from .load import parallel_scrna_gene_counter
from .concat import concatenate_exprs_matrices
from .doublet import predict_expression_doublets

log = getLogger('sneqtl')


def run_loadexprs(output_mat_dir, bam_fn, gtf_fn, cb_whitelist_fn,
                  min_mapq=255, cb_tag='CB', umi_tag='UB', umi_collapse_method='exact',
                  strand_type='fwd', quant_type='gene', processes=1):
    gene_ids, cb_whitelist, mtx = parallel_scrna_gene_counter(
        bam_fn, gtf_fn, cb_whitelist_fn,
        min_mapq=min_mapq,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        umi_dedup_method=umi_collapse_method,
        strand_filt_type=strand_type,
        quant_type=quant_type,
        njobs=processes
    )
    write_matrix_market(
        output_mat_dir, gene_ids, cb_whitelist, mtx
    )


def run_concatexprs(output_mat_dir, exprs_mat_dir, merge_suffixes=None):
    '''
    Concatenates expression matrices, potentially from different datasets, 
    adding suffixes to cell barcodes to avoid name collisions
    '''

    if merge_suffixes is None:
        merge_suffixes = [str(i + 1) for i in range(len(exprs_mat_dir))]

    if len(merge_suffixes) != len(exprs_mat_dir):
        raise ValueError('merge_suffixes must match number of mtx_dirs')

    mtx_objects = []
    for path in exprs_mat_dir:
        mtx = read_matrix_market_as_pd_matrix(path, as_dense=False)
        mtx_objects.append(mtx)
    concat_mtx = concatenate_exprs_matrices(mtx_objects, merge_suffixes, axis=1)
    concat_features = concat_mtx.index.tolist()
    concat_barcodes = concat_mtx.columns.tolist()
    write_matrix_market(
        output_mat_dir,
        concat_features,
        concat_barcodes,
        concat_mtx.sparse.to_coo()
    )


def run_doubletexprs(exprs_mat_dir, haplo_json_fn, cb_stats_fn, output_tsv_fn,
                     n_doublets=0.25, k_neighbours=100, min_cells_exprs=0.02,
                     min_pc_var_explained=0.01, max_pc_haplotype_var_explained=0.05,
                     celltype_n_clusters=None, cb_filter_exprs=None):
    if cb_stats_fn is not None:
        cb_stats = read_cb_stats(cb_stats_fn, cb_filter_exprs)
        cb_whitelist = list(cb_stats.index)
        log.info(f'Read {len(cb_whitelist)} barcodes from {cb_stats_fn}')
    else:
        cb_stats = None
        cb_whitelist = None
    exprs_mat = read_expression_matrix(
        exprs_mat_dir, cb_whitelist=cb_whitelist,
        rel_min_cells_exprs=min_cells_exprs,
    )
    log.info(f'Read expression matrix with {exprs_mat.shape[0]} genes and {exprs_mat.shape[1]} barcodes')

    haplotypes = PredictionRecords.read_json(haplo_json_fn)

    cb_whitelist = list(set(exprs_mat.columns).intersection(haplotypes.barcodes))
    if not len(cb_whitelist):
        raise ValueError('No matching barcodes between input files')
    exprs_mat = exprs_mat.loc[:, cb_whitelist]
    haplotypes = haplotypes.to_frame().loc[cb_whitelist]

    doublet_pred = predict_expression_doublets(
        exprs_mat, haplotypes,
        n_doublets=n_doublets,
        k_neighbours=k_neighbours,
        min_var_explained=min_pc_var_explained,
        max_pc_haplotype_var_explained=max_pc_haplotype_var_explained,
        n_clusters=celltype_n_clusters
    )
    if cb_stats is not None:
        cb_stats['expression_doublet_probability'] = doublet_pred
    else:
        cb_stats = doublet_pred.to_frame()
    log.info(f'Writing new stats to {output_tsv_fn}')
    cb_stats.to_csv(output_tsv_fn, sep='\t')
    return cb_stats