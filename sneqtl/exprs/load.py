import os
import re
from copy import copy
from collections import defaultdict, Counter
import itertools as it
from scipy import sparse
from scipy.io import mmwrite
import numpy as np

import pysam
from joblib import Parallel, delayed
import click

from snco.load.barcodes import read_cb_whitelist, umi_dedup_directional

from sneqtl.utils import parse_gtf
from snco.barcodes import umi_dedup_directional


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


def generate_scrna_count_matrix(genes, cb_whitelist, bam_fn, min_mapq=255,
                                strand_filt_type='fwd', cb_tag='CB', umi_tag='UB',
                                umi_dedup_method='exact'):
    gene_counts = sparse.dok_matrix((len(genes), len(cb_whitelist)), dtype=np.int16)
    cb_idx = {cb: i for i, cb in enumerate(cb_whitelist)}
    with pysam.AlignmentFile(bam_fn) as bam:
        for gene_idx, (gene_id, chrom, gene_strand, exons) in enumerate(genes):
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
                        umi = aln.get_tag(umi_tag)
                        try:
                            c_i = cb_idx[cb]
                        except KeyError:
                            continue
                        umi_counts[c_i][umi] += 1
                        gene_read_ids.add(read_id)

            for c_i, cb_umi_counts in umi_counts.items():
                if umi_dedup_method == 'directional':
                    cb_umi_counts = umi_dedup_directional(cb_umi_counts, has_haplotype=False)
                if umi_dedup_method == 'none':
                    count = cb_umi_counts.total()
                else:
                    count = len(cb_umi_counts)
                gene_counts[gene_idx, c_i] = count

    return gene_counts.tocsc()


def chunk_data(data, chunk_size=100):
    d_it = iter(data)
    for i in range(0, len(data), chunk_size):
        yield list(it.islice(d_it, chunk_size))


def parallel_scrna_gene_counter(gtf_fn, cb_whitelist_fn, bam_fn, quant_type='gene', njobs=1, **kwargs):
    cb_whitelist = read_cb_whitelist(cb_whitelist_fn).tolist()
    genes = parse_gtf(gtf_fn, model_type=quant_type)
    with Parallel(njobs, verbose=100) as pool:
        matrices = pool(
            delayed(generate_scrna_count_matrix)(
                gene_chunk, cb_whitelist, bam_fn, **kwargs
            ) for gene_chunk in list(chunk_data(genes))
        )
    gene_ids = [g[0] for g in genes]
    return gene_ids, cb_whitelist, sparse.vstack(matrices)


def write_features(feats_list, output_fn):
    with open(output_fn, 'w') as f:
        for feat in feats_list:
            f.write(f'{feat}\n')


def write_matrix_market(output_dir, gene_ids, cb_whitelist, mtx):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cb_fn = os.path.join(output_dir, 'barcodes.tsv')
    feat_fn = os.path.join(output_dir, 'features.tsv')
    mtx_fn = os.path.join(output_dir, 'matrix.mtx')
    write_features(cb_whitelist, cb_fn)
    write_features(gene_ids, feat_fn)
    mmwrite(mtx_fn, mtx)


def run_loadexprs(output_dir, gtf_fn, cb_whitelist_fn, bam_fn,
                  min_mapq=255, cb_tag='CB', umi_tag='UB', umi_dedup_method='exact',
                  strand_filt_type='fwd', quant_type='gene', processes=1):
    gene_ids, cb_whitelist, mtx = parallel_scrna_gene_counter(
        gtf_fn, cb_whitelist_fn, bam_fn,
        min_mapq=min_mapq,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        umi_dedup_method=umi_dedup_method,
        strand_filt_type=strand_filt_type,
        quant_type=quant_type,
        njobs=processes
    )

    write_matrix_market(
        output_dir, gene_ids, cb_whitelist, mtx
    )