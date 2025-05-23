import itertools as it

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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
