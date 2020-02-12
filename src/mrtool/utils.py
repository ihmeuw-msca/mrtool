# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~
    `utils` module of the `mrtool` package.
"""
import numpy as np
import pandas as pd


def get_cols(df, cols):
    """Return the columns of the given data frame.
    Args:
        df (pandas.DataFrame):
            Given data frame.
        cols (str | list{str} | None):
            Given column name(s), if is `None`, will return a empty data frame.
    Returns:
        pandas.DataFrame | pandas.Series:
            The data frame contains the columns.
    """
    assert isinstance(df, pd.DataFrame)
    if isinstance(cols, list):
        assert all([isinstance(col, str) and col in df
                    for col in cols])
    else:
        assert (cols is None) or (isinstance(cols, str) and cols in df)

    if cols is None:
        return df[[]]
    else:
        return df[cols]


def is_cols(cols):
    """Check variable type fall into the column name category.
    Args:
        cols (str | list{str} | None):
            Column names candidate.
    Returns:
        bool:
            if `col` is either str, list{str} or None
    """
    ok = isinstance(cols, (str, list)) or cols is None
    if isinstance(cols, list):
        ok = ok and all([isinstance(col, str)
                         for col in cols])
    return ok


def input_cols(cols, append_to=None, default=None):
    """Process the input column name.
    Args:
        cols (str | list{str} | None):
            The input column name(s).
        append_to (list{str} | None, optional):
            A list keep track of all the column names.
        default (str | list{str} | None, optional):
            Default value when `cols` is `None`.
    Returns:
        str | list{str}:
            The name of the column(s)
    """
    assert is_cols(cols)
    assert is_cols(append_to)
    assert is_cols(default)
    default = [] if default is None else default
    cols = default if cols is None else cols

    if isinstance(cols, list):
        cols = cols.copy()

    if cols is not None and append_to is not None:
        if isinstance(cols, str):
            append_to.append(cols)
        else:
            append_to += cols
    return cols


def combine_cols(cols):
    """Combine column names into one list of names.

    Args:
        cols (list{str | list{str}}):
            A list of names of columns or list of column names.

    Return:
        list{str}:
            Combined names of columns.
    """
    combined_cols = []
    for col in cols:
        if isinstance(col, str):
            combined_cols.append(col)
        else:
            combined_cols += col

    return combined_cols


def sizes_to_indices(sizes):
    """Converting sizes to corresponding indices.
    Args:
        sizes (numpy.dnarray):
            An array consist of non-negative number.
    Returns:
        list{range}:
            List the indices.
    """
    indices = []
    a = 0
    b = 0
    for i, size in enumerate(sizes):
        b += size
        indices.append(np.arange(a, b))
        a += size

    return indices
