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


def is_gaussian_prior(prior, size=None):
    """Check if variable satisfy Gaussian prior format

    Args:
        prior (numpy.ndarray):
            Either one or two dimensional array, with first group refer to mean
            and second group refer to standard deviation.

    Keyword Args:
        size (int | None, optional):
            Size the variable, prior related to.

    Returns:
        bool:
            True if satisfy condition.
    """
    # check type
    ok = isinstance(prior, np.ndarray) or prior is None
    if prior is not None:
        # check dimension
        ok = ok and (prior.ndim == 1 or prior.ndim == 2) and len(prior) == 2
        if size is not None and prior.ndim == 2:
            ok = ok and (prior.shape[1] == size)
        # check value
        ok = ok and np.all(prior[1] > 0.0)
    return ok


def is_uniform_prior(prior, size=None):
    """Check if variable satisfy uniform prior format

    Args:
        prior (numpy.ndarray):
            Either one or two dimensional array, with first group refer to lower
            bound and second group refer to upper bound.

    Keyword Args:
        size (int | None, optional):
            Size the variable, prior related to.

    Returns:
        bool:
            True if satisfy condition.
    """
    # check type
    ok = isinstance(prior, np.ndarray) or prior is None
    if prior is not None:
        # check dimension
        ok = ok and (prior.ndim == 1 or prior.ndim == 2) and len(prior) == 2
        if size is not None and prior.ndim == 2:
            ok = ok and (prior.shape[1] == size)
        # check value
        ok = ok and np.all(prior[0] <= prior[1])
    return ok


def input_gaussian_prior(prior, size):
    """Process the input Gaussian prior

    Args:
        prior (numpy.ndarray):
            Either one or two dimensional array, with first group refer to mean
            and second group refer to standard deviation.
        size (int, optional):
            Size the variable, prior related to.

    Returns:
        numpy.ndarray:
            Prior after processing, with shape (2, size), with the first row
            store the mean and second row store the standard deviation.
    """
    assert is_gaussian_prior(prior)
    if prior is None:
        return np.array([[0.0]*size, [np.inf]*size])
    elif prior.ndim == 1:
        return np.repeat(prior[:, None], size, axis=1)
    else:
        assert prior.shape[1] == size
        return prior

def input_uniform_prior(prior, size):
    """Process the input Gaussian prior

    Args:
        prior (numpy.ndarray):
            Either one or two dimensional array, with first group refer to mean
            and second group refer to standard deviation.
        size (int, optional):
            Size the variable, prior related to.

    Returns:
        numpy.ndarray:
            Prior after processing, with shape (2, size), with the first row
            store the mean and second row store the standard deviation.
    """
    assert is_gaussian_prior(prior)
    if prior is None:
        return np.array([[0.0]*size, [np.inf]*size])
    elif prior.ndim == 1:
        return np.repeat(prior[:, None], size, axis=1)
    else:
        assert prior.shape[1] == size
        return prior
