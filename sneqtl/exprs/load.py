import os
from logging import getLogger
import re
from copy import copy
from collections import defaultdict, Counter
from functools import reduce
import itertools as it
from scipy import sparse
from scipy.io import mmwrite
import numpy as np

import pysam
from joblib import Parallel, delayed
import click

from snco.load.barcodes import umi_dedup_directional
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.io.gtf import parse_gtf


log = getLogger('sneqtl')


def strand_filter(filt_type, gene_strand):
    gene_is_rev = gene_strand == '-'
    if filt_type == 'fwd':
        def _strand_filter(aln):
            return aln.is_reverse == gene_is_rev
    elif filt_type == 'rev':
        def _strand_filter(aln):
            return aln.is_reverse != gene_is_rev
    else:
        def _strand_filter(aln):
            return True
    return _strand_filter


def generate_scrna_count_dict(genes, bam_fn, cb_whitelist=None, min_mapq=255,
                              strand_filt_type='fwd', cb_tag='CB', umi_tag='UB',
                              umi_dedup_method='exact'):
    gene_counts = {}
    with pysam.AlignmentFile(bam_fn) as bam:
        for gene_id, chrom, gene_strand, exons in genes:
            sf = strand_filter(strand_filt_type, gene_strand)
            # umi dedup method
            umi_counts = defaultdict(Counter)
            gene_read_ids = set()
            for start, end in exons:
                for aln in bam.fetch(chrom, start, end):
                    read_id = aln.query_name
                    if read_id in gene_read_ids:
                        # prevents double counting of reads overlapping multiple exons
                        continue
                    if aln.mapq >= min_mapq and sf(aln):
                        cb = aln.get_tag(cb_tag)
                        if cb_whitelist is None or cb in cb_whitelist:
                            umi = aln.get_tag(umi_tag)
                            umi_counts[cb][umi] += 1
                            gene_read_ids.add(read_id)

            gene_counts[gene_id] = {}
            for cb, cb_umi_counts in umi_counts.items():
                if umi_dedup_method == 'directional':
                    cb_umi_counts = umi_dedup_directional(cb_umi_counts, has_haplotype=False)
                if umi_dedup_method == 'none':
                    count = cb_umi_counts.total()
                else:
                    count = len(cb_umi_counts)
                gene_counts[gene_id][cb] = count

    return gene_counts


def chunk_data(data, chunk_size=1000):
    d_it = iter(data)
    for i in range(0, len(data), chunk_size):
        yield list(it.islice(d_it, chunk_size))


def parallel_scrna_gene_counter(bam_fn, gtf_fn, cb_whitelist, quant_type='gene', njobs=1, **kwargs):
    genes = parse_gtf(gtf_fn, model_type=quant_type)
    log.info(f'Read {len(genes)} genes from "{gtf_fn}"')
    log.info(f'Processing genes with {njobs} processes')
    with Parallel(njobs) as pool:
        results = pool(
            delayed(generate_scrna_count_dict)(
                gene_chunk, bam_fn, cb_whitelist, **kwargs
            ) for gene_chunk in chunk_data(genes)
        )
    gene_counts = defaultdict(Counter)
    barcodes = set()
    for chunk_gene_counts in results:
        for gid, cb_dict in chunk_gene_counts.items():
            gene_counts[gid].update(cb_dict)
            barcodes.update(cb_dict)

    if cb_whitelist is not None:
        barcodes.intersection_update(cb_whitelist)
    barcodes = sorted(barcodes)
    log.info(f'Identified {len(barcodes)} barcodes with at least one UMI')
    final_cb_whitelist = set(barcodes)

    gene_ids = sorted([g[0] for g in genes])

    gene_ids = sorted([g[0] for g in genes])

    exprs_mat = ExpressionVariables.empty_sparse(
        index=gene_ids,
        columns=barcodes,
        sparse_type='dok',
        dtype=np.float32,
    )
    for gene_id, cb_counts in gene_counts.items():
        for cb, count in cb_counts.items():
            if cb in final_cb_whitelist:
                exprs_mat.loc[gene_id, cb] = count
    return exprs_mat
