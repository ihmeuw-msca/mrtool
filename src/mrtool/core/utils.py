# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~
    `utils` module of the `mrtool` package.
"""
from typing import List
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


def sample_knots(num_knots: int = 5,
                 sample_width: float = 0.05,
                 sample_size: int = 1) -> np.ndarray:
    """
    Sample knots

    Parameters
    ----------
    num_knots : int, optional
        Total number of knots, include the boundary knots, must be greater or
        equal than 2. Default to be 5.
    sample_width : float, optional
        Sample interval for each knot. For example, if ``num_knots=3``, we have
        one interior knots and the average position is 0.5.
        If ``sample_width=0.1``, we sample the interior knot between,
        ``[0.5 - 0.1, 0.5 + 0.1]``. Default to be 0.05.
    sample_size : int, optional
        Number of samples, must be greater or equal to 1. Default to be 1.

    Returns
    -------
    ndarray
        Knots samples matrix, with shape ``(sample_size, num_knots)``.
    """
    if num_knots < 2:
        raise ValueError("Number of knots must be greater or equal than 2.")
    if sample_width < 0.0 or sample_width > 1.0:
        raise ValueError("sample_width need to be between 0 and 1.")
    if sample_size < 1:
        raise ValueError("sample_size at least need to be 1.")

    knots = np.linspace(0, 1, int(num_knots))
    inner_knots = knots[1:-1]
    if inner_knots.size == 0:
        return np.tile(knots, (sample_size, 1))

    inner_knots_bounds = np.vstack([
        np.minimum(1.0, np.maximum(0.0, inner_knots - sample_width)),
        np.minimum(1.0, np.maximum(0.0, inner_knots + sample_width))
    ])

    inner_knots_samples = np.random.uniform(
        inner_knots_bounds[0],
        inner_knots_bounds[1],
        size=(sample_size, inner_knots.size)
    )

    knots_samples = np.hstack([
        np.zeros((sample_size, 1)),
        inner_knots_samples,
        np.ones((sample_size, 1))
    ])
    knots_samples.sort(axis=1)

    return knots_samples
