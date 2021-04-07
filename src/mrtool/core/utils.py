# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~
    `utils` module of the `mrtool` package.
"""
from typing import List, Any, Tuple
import numpy as np


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
    for size in sizes:
        b += size
        indices.append(np.arange(a, b))
        a += size

    return indices


def sizes_to_slices(sizes: np.array) -> List[slice]:
    ends = np.cumsum(sizes)
    starts = np.insert(ends, 0, 0)[:-1]
    return [slice(*pair) for pair in zip(starts, ends)]


def avg_integral(mat, spline=None):
    """Compute average integral.

    Args:
        mat (numpy.ndarray):
            Matrix that contains the starting and ending points of the integral
            or a single column represents the mid-points.
        spline (xspline.XSpline | None, optional):
            Spline integrate over with, when `None` treat the function as
            linear.

    Returns:
        numpy.ndarray:
            Design matrix when spline is not `None`, otherwise the mid-points.
    """
    assert mat.ndim == 2
    if mat.size == 0:
        return mat.reshape(mat.shape[0], 0)

    if mat.shape[1] == 1:
        return mat if spline is None else spline.design_mat(
            mat.ravel(), l_extra=True, r_extra=True)
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

            return mat


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


def proj_simplex(x: np.ndarray) -> np.ndarray:
    # sort x in the desending order
    u = x.copy()
    u[::-1].sort()

    # compute intermediate variable
    j = np.arange(u.size)
    v = (1.0 - np.cumsum(u))/(j + 1)

    # pick index and compute lambda
    rho = np.max(j*(u + v > 0))
    lam = v[rho]

    return np.maximum(x + lam, 0.0)
