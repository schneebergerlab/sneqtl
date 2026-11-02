import logging

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pyarrow import dataset as ds
from joblib import Parallel, delayed

from .io.gtf import read_gtf_gene_locs
from .eqtl.model.qtl import SingleGeneResults
from .eqtl.analysis import eQTLResults

log = logging.getLogger('sneqtl')


def lod_peak_finder(lod_scores, pvals, abs_coefs, lod_threshold, pval_threshold,
                    rel_lod_threshold, coef_abs_threshold,
                    rel_prominence, ci_lod_drop, min_dist):
    drop_threshold = max(max(lod_scores) - ci_lod_drop, 0)
    # insert infs at beginning and end of chromosome to prevent prominence edge effects
    lod_scores_padded = np.insert(lod_scores, [0, len(lod_scores)], [-np.inf, -np.inf])
    peak_idx, _ = find_peaks(
        lod_scores_padded,
        height=max(lod_threshold, max(lod_scores) * rel_lod_threshold),
        prominence=max(lod_scores) * rel_prominence,
        distance=min_dist
    )
    peaks = []
    for p_idx in peak_idx:
        if pvals[p_idx - 1] > pval_threshold:
            continue
        if abs_coefs[p_idx - 1] < coef_abs_threshold:
            continue
        l_ci_idx = p_idx
        while (lod_scores_padded[l_ci_idx - 1] > drop_threshold):
            l_ci_idx -= 1
        r_ci_idx = p_idx
        while (lod_scores_padded[r_ci_idx + 1] > drop_threshold):
            r_ci_idx += 1
        # shift idx due to padding
        yield p_idx - 1, l_ci_idx - 1, r_ci_idx - 1


def _make_array(bin_starts, bin_size, vals, fill_val=0.0):
    i = bin_starts // bin_size
    out = np.full(i.max() + 1, fill_val)
    out[i] = vals
    return out


def call_eqtl_peaks_gene(gene_eqtl_results, lod_var='lod_score', pval_var='lrt_pval', coef_var=None,
                         lod_threshold=5, rel_lod_threshold=0.1, coef_abs_threshold=0.1,
                         rel_prominence=0.25, ci_lod_drop=1.5,
                         min_dist_between_eqtls=3e6, bin_size=25_000,
                         snps=False):
    min_dist = int(min_dist_between_eqtls) // int(bin_size)

    if pval_var == 'lrt_pval':
        pval_threshold = gene_eqtl_results.attrs['lrt_pval_threshold']
    else:
        pval_threshold = gene_eqtl_results.attrs['param_pval_thresholds'][pval_var]

    eqtl_peaks = []
    for _, chrom_eqtl_results in gene_eqtl_results.groupby('chrom'):
        if snps:
            # filter for best snp cluster per bin
            chrom_eqtl_results = chrom_eqtl_results.loc[
                chrom_eqtl_results.groupby('pos').lod_score.idxmax()
            ]
        chrom_eqtl_results = chrom_eqtl_results.sort_values('pos').set_index('pos', drop=False)
        bin_starts = chrom_eqtl_results.index.values
        if coef_var is not None:
            coefs = chrom_eqtl_results[[coef_var,]]
        else:
            coefs = chrom_eqtl_results[[c for c in chrom_eqtl_results.columns if c.endswith('coef')]]
        abs_coef = _make_array(bin_starts, bin_size, coefs.abs().max(axis=1).values, 0.0)
        lods = _make_array(bin_starts, bin_size, chrom_eqtl_results[lod_var].values, 0.0)
        pvals = _make_array(bin_starts, bin_size, chrom_eqtl_results[pval_var].values, 1.0)
        lod_peaks = lod_peak_finder(
            lods, pvals,
            abs_coefs=abs_coef,
            lod_threshold=lod_threshold,
            pval_threshold=pval_threshold,
            rel_lod_threshold=rel_lod_threshold,
            coef_abs_threshold=coef_abs_threshold,
            rel_prominence=rel_prominence,
            ci_lod_drop=ci_lod_drop,
            min_dist=min_dist
        )
        for peak_idx, left_ci_idx, right_ci_idx in lod_peaks:
            e = chrom_eqtl_results.loc[peak_idx * bin_size].copy()
            e['left_ci_eqtl'] = left_ci_idx * bin_size
            e['right_ci_eqtl'] = (right_ci_idx + 1) * bin_size
            eqtl_peaks.append(e)
    eqtl_peaks = pd.DataFrame(eqtl_peaks)
    return eqtl_peaks.reset_index(drop=True)


def call_eqtl_peaks(eqtl_results, gene_locs, peak_variable='overall', cis_range=5e5, **kwargs):
    lod_threshold = kwargs.get('lod_threshold', 5)
    snps = kwargs.get('snps', False)
    if peak_variable == 'overall':
        lod_var = 'lod_score'
        pval_var = 'lrt_pval'
    else:
        lod_var = f'{peak_variable}_lod_score'
        pval_var = f'{peak_variable}_pval'
        if not lod_var in eqtl_results.schema.names or not pval_var in eqtl_results.schema.names:
            raise ValueError(f'Variable "{peak_variable}" was not tested')

    def lod_filt(gene_eqtl_results):
        pval_threshold = (gene_eqtl_results.attrs['lrt_pval_threshold'] if pval_var == 'lrt_pval'
                          else gene_eqtl_results.attrs['param_pval_thresholds'][pval_var])
        return (gene_eqtl_results[lod_var].max() > lod_threshold) & \
               (gene_eqtl_results[pval_var].min() < pval_threshold)

    eqtl_peaks = []
    for gene_id, gene_results in eqtl_results.iter_results():
        gene_results = gene_results.to_dataframe()
        if lod_filt(gene_results):
            eqtl_peaks.append(call_eqtl_peaks_gene(gene_results, lod_var=lod_var, pval_var=pval_var, **kwargs))

    if not eqtl_peaks:
        eqtl_peaks = pd.DataFrame([], columns=[
            'chrom', 'pos', 'left_ci_eqtl', 'right_ci_eqtl',
            'gene_id', 'lod_score', 'lrt_pval',
            *eqtl_results.schema.names[5:]
        ])
    else:
        eqtl_peaks = pd.concat(eqtl_peaks, axis=0).reset_index(drop=True)
    eqtl_peaks = pd.merge(
        eqtl_peaks,
        gene_locs,
        left_on='gene_id',
        right_index=True,
        how='left',
        suffixes=['_eqtl', '_gene']
    )
    eqtl_peaks['eqtl_type'] = eqtl_peaks.eval(
        'chrom_eqtl == chrom_gene & '
        '(left_ci_eqtl - @cis_range) <= pos_gene <= (right_ci_eqtl + @cis_range)'
    ).map({True: 'cis', False: 'trans'})
    columns = [
        'chrom_eqtl', 'pos_eqtl', 'left_ci_eqtl', 'right_ci_eqtl',
        'gene_id', 'chrom_gene', 'pos_gene', 'eqtl_type',
        'lod_score', 'lrt_pval',
        *eqtl_results.schema.names[6 if snps else 5:]
    ]
    if kwargs['snps']:
        columns.insert(4, 'snp_id')
    return eqtl_peaks[columns]


def run_peakcall(eqtl_dir, output_tsv_fn, gtf_fn, *, eqtl_results=None,
                 peak_variable='overall', lod_threshold=5, rel_lod_threshold=0.1,
                 rel_prominence=0.25, ci_lod_drop=1.5,
                 min_dist_between_eqtls=3e6, cis_eqtl_range=5e5):
    if eqtl_results is None:
        eqtl_results = eQTLResults(eqtl_dir)
    bin_size = int(eqtl_results.schema.field("pos").metadata[b"bin_size"].decode())
    is_snp_based = 'snp_id' in eqtl_results.schema.names
    gene_locs = read_gtf_gene_locs(gtf_fn)
    log.info(f'Calling eQTL peaks for {len(eqtl_results)} genes')
    eqtl_peaks = call_eqtl_peaks(
        eqtl_results, gene_locs,
        peak_variable=peak_variable,
        lod_threshold=lod_threshold,
        rel_lod_threshold=rel_lod_threshold,
        rel_prominence=rel_prominence,
        ci_lod_drop=ci_lod_drop,
        min_dist_between_eqtls=min_dist_between_eqtls,
        cis_range=cis_eqtl_range,
        snps=is_snp_based,
    )
    if output_tsv_fn is not None:
        log.info(f'Writing peaks to {output_tsv_fn}')
        eqtl_peaks.to_csv(
            output_tsv_fn,
            sep='\t',
            index=False,
            header=True,
            float_format='%.4g'
        )
    return eqtl_peaks