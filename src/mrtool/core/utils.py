# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~
    `utils` module of the `mrtool` package.
"""
from typing import Union, List, Any, Tuple
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

is_laplace_prior = is_gaussian_prior

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
    if prior is None or prior.size == 0:
        return np.array([[0.0]*size, [np.inf]*size])
    elif prior.ndim == 1:
        return np.repeat(prior[:, None], size, axis=1)
    else:
        assert prior.shape[1] == size
        return prior

input_laplace_prior = input_gaussian_prior

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
    assert is_uniform_prior(prior)
    if prior is None or prior.size == 0:
        return np.array([[-np.inf]*size, [np.inf]*size])
    elif prior.ndim == 1:
        return np.repeat(prior[:, None], size, axis=1)
    else:
        assert prior.shape[1] == size
        return prior


def avg_integral(mat, spline=None, use_spline_intercept=False):
    """Compute average integral.

    Args:
        mat (numpy.ndarray):
            Matrix that contains the starting and ending points of the integral
            or a single column represents the mid-points.
        spline (xspline.XSpline | None, optional):
            Spline integrate over with, when `None` treat the function as
            linear.
        use_spline_intercept (bool, optional):
            If `True` use all bases from spline, otherwise remove the first bases.

    Returns:
        numpy.ndarray:
            Design matrix when spline is not `None`, otherwise the mid-points.
    """
    assert mat.ndim == 2
    if mat.size == 0:
        return mat.reshape(mat.shape[0], 0)

    index = 0 if use_spline_intercept else 1

    if mat.shape[1] == 1:
        return mat if spline is None else spline.design_mat(
            mat.ravel(), l_extra=True, r_extra=True)[:, index:]
    else:
        if spline is None:
            return mat.mean(axis=1)[:, None]
        else:
            x0 = mat[:, 0]
            x1 = mat[:, 1]
            dx = x1 - x0
            val_idx = (dx == 0.0)
            int_idx = (dx != 0.0)

            mat = np.zeros((dx.size, spline.num_spline_bases))

            if np.any(val_idx):
                mat[val_idx, :] = spline.design_mat(x0[val_idx],
                                                    l_extra=True,
                                                    r_extra=True)
            if np.any(int_idx):
                mat[int_idx, :] = spline.design_imat(
                    x0[int_idx], x1[int_idx], 1,
                    l_extra=True,
                    r_extra=True)/(dx[int_idx][:, None])

            return mat[:, index:]


# random knots
def sample_knots(num_knots: int,
                 knot_bounds: np.ndarray,
                 min_dist: Union[float, np.ndarray],
                 num_samples: int = 1) -> np.ndarray:
    """Sample knot vectors given a set of rules.

    Parameters
    ----------
    num_knots : int
        Number of interior knots.
    knot_bounds : np.ndarray, shape(2,) or shape(`num_knots`,2)
        Lower and upper bounds for knots. If shape(2,), boundary knots
        placed at `knot_bounds[0]` and `knot_bounds[1]`. If
        shape(`num_knots`,2), boundary knots placed at
        `knot_bounds[0, 0]` and `knot_bounds[-1, 1]`.
    min_dist : float or np.ndarray, shape(`num_knots`+1,)
        Minimum distances between knots.
    num_samples : int, optional
        Number of knot vectors to sample. Default is 1.

    Returns
    -------
    np.ndarray, shape(`num_samples`,`num_knots`+2)
        Sampled knot vectors.

    """
    # Check num_knots and num_samples
    num_dict = {'num_knots': num_knots, 'num_samples': num_samples}
    for num_name, num_val in num_dict.items():
        if not isinstance(num_val, int):
            raise TypeError(f"{num_name} must be an integer")
        if num_val < 1:
            raise ValueError(f"{num_name} must be at least 1")

    # Check knot_bounds
    if not isinstance(knot_bounds, np.ndarray):
        raise TypeError('knot_bounds must be an array')
    if knot_bounds.shape != (num_knots, 2):
        if knot_bounds.shape == (2,):
            knot_bounds = np.tile(knot_bounds, (num_knots, 1))
        else:
            msg = 'knot_bounds must have shape (2,) or (num_knots,2)'
            raise ValueError(msg)
    bounds_sorted = np.all(np.diff(knot_bounds, axis=1) > 0.)
    neighbors_sorted = np.all(np.diff(knot_bounds, axis=0) >= 0.)
    if not (bounds_sorted and neighbors_sorted):
        raise ValueError('knot_bounds must be sorted')

    # Check min_dist
    if not isinstance(min_dist, np.ndarray):
        if isinstance(min_dist, float):
            min_dist = np.tile(min_dist, num_knots + 1)
        else:
            raise TypeError('min_dist must be a float or array')
    if min_dist.shape != (num_knots + 1,):
        raise ValueError('min_dist must have shape(num_knots+1,)')
    if np.any(min_dist < 0.):
        raise ValueError('min_dist must be positive')
    if np.sum(min_dist) > knot_bounds[-1, 1] - knot_bounds[0, 0]:
        raise ValueError('min_dist cannot exceed knot_bounds')

    # Check feasibility
    left_bounds = np.zeros(num_knots)
    left_bounds[0] = knot_bounds[0, 0] + min_dist[0]
    for ii in range(1, num_knots):
        left_bounds[ii] = np.maximum(knot_bounds[ii, 0],
                                     left_bounds[ii - 1] + min_dist[ii])
    right_bounds = np.zeros(num_knots)
    right_bounds[-1] = knot_bounds[-1, 1] - min_dist[-1]
    for ii in range(-2, -(num_knots + 1), -1):
        right_bounds[ii] = np.minimum(knot_bounds[ii, 1],
                                      right_bounds[ii + 1] - min_dist[ii])
    if np.any(left_bounds > right_bounds):
        raise ValueError('knot_bounds and min_dist not feasible')

    # Sample knots
    knots = np.zeros((num_samples, num_knots + 2))
    knots[:, 0] = knot_bounds[0, 0]
    knots[:, -1] = knot_bounds[-1, 1]
    for ii in range(num_knots):
        left_bound = np.maximum(left_bounds[ii], knots[:, ii] + min_dist[ii])
        if np.any(left_bound > right_bounds[ii]):
            raise ValueError('empty sampling interval')
        knots[:, ii + 1] = np.random.uniform(left_bound, right_bounds[ii])

    return knots


def sample_simplex(n, N=1):
    """sample from n dimensional simplex"""
    assert n >= 1

    # special case when n == 1
    if n == 1:
        return np.ones((N, n))

    # other cases
    s = np.random.rand(N, n - 1)
    s.sort(axis=1)
    s = np.insert(s, 0, 0.0, axis=1)
    s = np.insert(s, n, 1.0, axis=1)

    w = np.zeros((n + 1, n))
    id_d0 = np.diag_indices(n)
    id_d1 = (id_d0[0] + 1, id_d0[1])
    w[id_d0] = -1.0
    w[id_d1] = 1.0

    return s.dot(w)


def col_diff_mat(n):
    """column difference matrix"""
    D = np.zeros((n + 1, n))
    id_d0 = np.diag_indices(n)
    id_d1 = (id_d0[0] + 1, id_d0[1])
    D[id_d0] = -1.0
    D[id_d1] = 1.0

    return D


def nonlinear_trans(score, slope=6.0, quantile=0.7):
    score_min = np.min(score)
    score_max = np.max(score)
    if score_max == score_min:
        return np.ones(len(score))
    else:
        weight = (score - score_min)/(score_max - score_min)

    sorted_weight = np.sort(weight)
    x = sorted_weight[int(0.8*weight.size)]
    y = 1.0 - x

    # calculate the transformation coefficient
    c = np.zeros(4)
    c[1] = slope*x**2/quantile
    c[0] = quantile*np.exp(c[1]/x)
    c[3] = slope*y**2/(1.0 - quantile)
    c[2] = (1.0 - quantile)*np.exp(c[3]/y)

    weight_trans = np.zeros(weight.size)

    for i in range(weight.size):
        w = weight[i]
        if w == 0.0:
            weight_trans[i] = 0.0
        elif w < x:
            weight_trans[i] = c[0]*np.exp(-c[1]/w)
        elif w < 1.0:
            weight_trans[i] = 1.0 - c[2]*np.exp(-c[3]/(1.0 - w))
        else:
            weight_trans[i] = 1.0

    weight_trans = (weight_trans - np.min(weight_trans))/\
        (np.max(weight_trans) - np.min(weight_trans))

    return weight_trans


def mat_to_fun(alt_mat, ref_mat=None):
    alt_mat = np.array(alt_mat)
    assert alt_mat.ndim == 2
    if ref_mat is not None:
        ref_mat = np.array(ref_mat)
        assert ref_mat.ndim == 2

    if alt_mat.size == 0:
        fun = None
        jac_fun = None
    else:
        if ref_mat is None or ref_mat.size == 0:
            mat = alt_mat
        else:
            mat = alt_mat - ref_mat
        def fun(x, mat=mat):
            return mat.dot(x)

        def jac_fun(x, mat=mat):
            return mat

    return fun, jac_fun

def mat_to_log_fun(alt_mat, ref_mat=None, add_one=True):
    alt_mat = np.array(alt_mat)
    shift = 1.0 if add_one else 0.0
    assert alt_mat.ndim == 2
    if ref_mat is not None:
        ref_mat = np.array(ref_mat)
        assert ref_mat.ndim == 2

    if alt_mat.size == 0:
        fun = None
        jac_fun = None
    else:
        if ref_mat is None or ref_mat.size == 0:
            def fun(beta):
                return np.log(shift + alt_mat.dot(beta))

            def jac_fun(beta):
                return alt_mat/(shift + alt_mat.dot(beta)[:, None])
        else:
            def fun(beta):
                return np.log(shift + alt_mat.dot(beta)) - \
                       np.log(shift + ref_mat.dot(beta))

            def jac_fun(beta):
                return alt_mat/(shift + alt_mat.dot(beta)[:, None]) - \
                       ref_mat/(shift + ref_mat.dot(beta)[:, None])

    return fun, jac_fun


def empty_array():
    return np.array(list())

def to_list(obj: Any) -> List[Any]:
    """Convert objective to list of object.

    Args:
        obj (Any): Object need to be convert.

    Returns:
        List[Any]:
            If `obj` already is a list object, return `obj` itself,
            otherwise wrap `obj` with a list and return it.
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def is_numeric_array(array: np.ndarray) -> bool:
    """Check if an array is numeric.

    Args:
        array (np.ndarray): Array need to be checked.

    Returns:
        bool: True if the array is numeric.
    """
    numerical_dtype_kinds = {'b',  # boolean
                             'u',  # unsigned integer
                             'i',  # signed integer
                             'f',  # floats
                             'c'}  # complex
    try:
        return array.dtype.kind in numerical_dtype_kinds
    except AttributeError:
        # in case it's not a numpy array it will probably have no dtype.
        return np.asarray(array).dtype.kind in numerical_dtype_kinds


def expand_array(array: np.ndarray,
                 shape: Tuple[int],
                 value: Any,
                 name: str) -> np.ndarray:
    """Expand array when it is empty.

    Args:
        array (np.ndarray):
            Target array. If array is empty, fill in the ``value``. And
            When it is not empty assert the ``shape`` agrees and return the original array.
        shape (Tuple[int]): The expected shape of the array.
        value (Any): The expected value in final array.
        name (str): Variable name of the array (for error message).

    Returns:
        np.ndarray: Expanded array.
    """
    array = np.array(array)
    if len(array) == 0:
        if hasattr(value, '__iter__') and not isinstance(value, str):
            value = np.array(value)
            assert value.shape == shape, f"{name}, alternative value inconsistent shape."
            array = value
        else:
            array = np.full(shape, value)
    else:
        assert array.shape == shape, f"{name}, inconsistent shape."
    return array


def ravel_dict(x: dict) -> dict:
    """Ravel dictionary.
    """
    assert all([isinstance(k, str) for k in x.keys()])
    assert all([isinstance(v, np.ndarray) for v in x.values()])
    new_x = {}
    for k, v in x.items():
        if v.size == 1:
            new_x[k] = v
        else:
            for i in range(v.size):
                new_x[f'{k}_{i}'] = v[i]
    return new_x