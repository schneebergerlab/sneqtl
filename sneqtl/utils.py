import os
import re
import logging
import itertools as it

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.io import mmread

log = logging.getLogger('snco')


def whitelist_filter(df, whitelist):
    if df is None or df.empty:
        return pd.DataFrame()
    else:
        return df.loc[whitelist]


def get_dummies(df, **kwargs):
    if df is None or df.empty:
        return pd.DataFrame()
    dummies = pd.get_dummies(df, **kwargs).astype('float')
    return dummies


def convert_to_numeric(df, drop_first=False):
    converted = []
    for colname, col in df.items():
        if is_numeric_dtype(col):
            converted.append(col)
        else:
            converted.append(
                pd.get_dummies(col, drop_first=drop_first, prefix=colname).astype(int)
            )
    return pd.concat(converted, axis=1)


def drop_first_column(df):
    if df is None or df.empty:
        return pd.DataFrame()
    return df.iloc[:, 1:]


def aligned_concat_axis1(df_list, as_df=True):
    concat_data = np.concatenate([df.values for df in df_list], axis=1)
    if as_df:
        columns = list(it.chain(*(df.columns.tolist() for df in df_list)))
        index = df_list[0].index
        concat_data = pd.DataFrame(concat_data, index=index, columns=columns)
    return concat_data


def concat_handle_none(df_list, **kwargs):
    df_list = [df for df in df_list if (df is not None and not df.empty)]
    if not len(df_list):
        return pd.DataFrame()
    else:
        return pd.concat(df_list, **kwargs)


def expanding_mul(df1, df2, join_char='_'):
    product =  pd.concat({
        colname: df2.mul(col, axis=0)
        for colname, col in df1.items()
    }, axis=1)
    product.columns = [join_char.join(colname).strip() for colname in product.columns.values]
    return product


def merge_sum_overlapping_columns(dfs):
    df = pd.concat(dfs, axis=1)
    return df.T.groupby(level=0).sum().T


def align_input(exprs_mat, haplotypes, cb_whitelist=None, covariates=None,
                interacting_variables=None, parental_genotypes=None):
    if cb_whitelist is None:
        cb_whitelist = list(set(exprs_mat.columns).intersection(haplotypes.index))
        if not len(cb_whitelist):
            raise ValueError('No overlap between exprs_mat, haplotypes and covariates')
    exprs_mat = exprs_mat.loc[:, cb_whitelist]
    haplotypes = haplotypes.loc[cb_whitelist]

    covariates = whitelist_filter(covariates, cb_whitelist)
    interacting_variables = whitelist_filter(interacting_variables, cb_whitelist)
    parental_genotypes = whitelist_filter(parental_genotypes, cb_whitelist)
    return exprs_mat, haplotypes, covariates, interacting_variables, parental_genotypes


def read_cb_stats(cb_stats_fn, cb_filter_exprs=None):
    cb_stats = pd.read_csv(
        cb_stats_fn, index_col=0, sep='\t',
    )
    if cb_filter_exprs is not None:
        cb_stats = cb_stats.query(cb_filter_exprs)
    return cb_stats


def get_gtf_attribute(gtf_record, attribute):
    try:
        attr = re.search(f'{attribute} "(.+?)";', gtf_record[8]).group(1)
    except AttributeError:
        raise ValueError(
            f'Could not parse attribute {attribute} '
            f'from GTF with feature type {record[2]}'
        )
    return attr


def flatten_intervals(invs):
    flattened = []
    inv_it = iter(invs)
    inv_start, inv_end = next(inv_it)
    for start, end in inv_it:
        if start <= inv_end:
            inv_end = max(inv_end, end)
        else:
            flattened.append([inv_start, inv_end])
            inv_start, inv_end = start, end
    if not flattened or flattened[-1] != [inv_start, inv_end]:
        flattened.append([inv_start, inv_end])
    return flattened


def parse_gtf(gtf_fn, model_type='gene'):
    gtf_records = {}
    with open(gtf_fn) as gtf:
        for i, record in enumerate(gtf):
            if record.startswith('#'):
                continue
            record = record.split('\t')
            chrom, _, feat_type, start, end, _, strand = record[:7]
            start = int(start) - 1
            end = int(end)
            if feat_type == 'exon':
                gene_id = get_gtf_attribute(record, 'gene_id')
                idx = (chrom, strand, gene_id)
                if idx not in gtf_records:
                    gtf_records[idx] = []
                gtf_records[idx].append([start, end])
    gene_records = []
    for (chrom, strand, gene_id), invs in gtf_records.items():
        invs.sort()
        if model_type == 'gene':
            invs = flatten_intervals(invs)
        elif model_type == 'gene_full':
            invs = [[invs[0][0], invs[-1][1]]]
        else:
            raise ValueError('unrecognised quant_type')
        gene_records.append((gene_id, chrom, strand, invs))
    return gene_records


def read_gtf_gene_locs(gtf_fn):
    gene_locs = {}
    for gene_id, chrom, strand, invs in  parse_gtf(gtf_fn, model_type='gene_full'):
        gene_locs[gene_id] = (chrom, (invs[0][0] + invs[-1][1]) // 2)
    return pd.DataFrame.from_dict(gene_locs, orient='index', columns=['chrom', 'pos'])


def read_eqtl_results(eqtl_res_fn):
    eqtl_results = pd.read_csv(
        eqtl_res_fn,
        sep='\t',
        dtype={
            'chrom': str,
            'pos': int,
            'lod_score': float,
            'pval': float,
        }
    )
    test_gene_id, test_chrom = eqtl_results.iloc[0].loc[['gene_id', 'chrom']]
    test_pos = eqtl_results.query('gene_id == @test_gene_id & chrom == @test_chrom').pos.values
    bin_size = int(np.unique(np.diff(np.sort(test_pos))))
    return eqtl_results, bin_size