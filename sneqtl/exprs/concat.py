import pandas as pd


def concatenate_exprs_matrices(mtxs, merge_suffixes=None, axis=0):
    """
    Concatenates multiple pandas sparse DataFrames.

    Parameters
    ----------
    mtxs : list
        List of pandas sparse DataFrames to concatenate.
    merge_suffixes : list or None, optional
        List of suffixes to append to the index (if axis=0) or columns (if axis=1) 
        of each DataFrame. If None, a range of integers (1, 2, ..., len(mtxs)) is used.
    axis : int
        The axis to concatenate along: 0 for rows, 1 for columns.

    Returns
    -------
    pandas.DataFrame
        A single sparse DataFrame representing the concatenated result.

    Raises
    ------
    ValueError
        If the number of provided suffixes does not match the number of DataFrames, or
        if the inputs are not all pandas sparse DataFrames.
    """
    if not mtxs:
        raise ValueError('No DataFrames provided')

    if not all(isinstance(df, pd.DataFrame) for df in mtxs):
        raise ValueError('All inputs must be pandas DataFrames')

    if not all(df.dtypes.apply(pd.api.types.is_sparse).all() for df in mtxs):
        raise ValueError('All DataFrames must have only sparse columns')

    if merge_suffixes is None:
        merge_suffixes = [str(i + 1) for i in range(len(mtxs))]

    if len(merge_suffixes) != len(mtxs):
        raise ValueError('Number of suffixes must match number of DataFrames')

    renamed_mtxs = []
    for df, suffix in zip(mtxs, merge_suffixes):
        df_copy = df.copy()
        if axis == 0:
            df_copy.index = df_copy.index.astype(str) + f'_{suffix}'
        else:
            df_copy.columns = df_copy.columns.astype(str) + f'_{suffix}'
        renamed_mtxs.append(df_copy)

    result = pd.concat(renamed_mtxs, axis=axis)
    return result