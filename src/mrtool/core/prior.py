"""
Prior module
"""
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
from numpy import ndarray
from xspline import XSpline

from regmod.utils import default_vec_factory

default_params = {
    "Gaussian": np.array([[0.0], [np.inf]]),
    "Uniform": np.array([[-np.inf], [np.inf]])
}

default_prior_params = {
    "gprior": default_params["Gaussian"],
    "uprior": default_params["Uniform"],
    "linear_gprior": default_params["Gaussian"],
    "linear_uprior": default_params["Uniform"]
}


@dataclass
class Prior:
    size: int = None
    ptype: str = field(default="prior", repr=False, init=False)

    def process_size(self, vecs: List[Any]):
        if self.size is None:
            sizes = [len(vec) for vec in vecs if isinstance(vec, Iterable)]
            sizes.append(1)
            self.size = max(sizes)

        if self.size <= 0 or not isinstance(self.size, int):
            raise ValueError("Size of the prior must be a positive integer.")


@dataclass
class GaussianPrior(Prior):
    mean: ndarray = field(default=None, repr=False)
    sd: ndarray = field(default=None, repr=False)
    ptype = "gprior"

    def __post_init__(self):
        self.process_size([self.mean, self.sd])
        self.mean = default_vec_factory(self.mean, self.size, 0.0, vec_name='mean')
        self.sd = default_vec_factory(self.sd, self.size, np.inf, vec_name='sd')
        assert all(self.sd > 0.0), "Standard deviation must be all positive."

    @property
    def info(self) -> ndarray:
        return np.vstack([self.mean, self.sd])


@dataclass
class UniformPrior(Prior):
    lb: ndarray = field(default=None, repr=False)
    ub: ndarray = field(default=None, repr=False)
    ptype = "uprior"

    def __post_init__(self):
        self.process_size([self.lb, self.ub])
        self.lb = default_vec_factory(self.lb, self.size, -np.inf, vec_name='lb')
        self.ub = default_vec_factory(self.ub, self.size, np.inf, vec_name='ub')
        assert all(self.lb <= self.ub), "Lower bounds must be less or equal than upper bounds."

    @property
    def info(self) -> ndarray:
        return np.vstack([self.lb, self.ub])


@dataclass
class LinearPrior:
    mat: ndarray = field(default_factory=lambda: np.empty(shape=(0, 1)),
                         repr=False)
    size: int = None
    ptype = "linear_prior"

    def __post_init__(self):
        if self.size is None:
            self.size = self.mat.shape[0]
        else:
            assert self.size == self.mat.shape[0], "`mat` and `size` not match."

    def is_empty(self) -> bool:
        return self.mat.size == 0.0


@dataclass
class SplinePrior(LinearPrior):
    size: int = 100
    order: int = 0
    domain_lb: float = field(default=0.0, repr=False)
    domain_ub: float = field(default=1.0, repr=False)
    domain_type: str = field(default="rel", repr=False)

    def __post_init__(self):
        assert self.domain_lb <= self.domain_ub, \
            "Domain lower bound must be less or equal than upper bound."
        assert self.domain_type in ["rel", "abs"], \
            "Domain type must be 'rel' or 'abs'."
        if self.domain_type == "rel":
            assert self.domain_lb >= 0.0 and self.domain_ub <= 1.0, \
                "Using relative domain, bounds must be numbers between 0 and 1."

    def attach_spline(self, spline: XSpline) -> ndarray:
        knots_lb = spline.knots[0]
        knots_ub = spline.knots[-1]
        if self.domain_type == "rel":
            points_lb = knots_lb + (knots_ub - knots_lb)*self.domain_lb
            points_ub = knots_lb + (knots_ub - knots_lb)*self.domain_ub
        else:
            points_lb = self.domain_lb
            points_ub = self.domain_ub
        points = np.linspace(points_lb, points_ub, self.size)
        self.mat = spline.design_dmat(points, order=self.order,
                                      l_extra=True, r_extra=True)
        super().__post_init__()


@dataclass
class LinearGaussianPrior(LinearPrior, GaussianPrior):
    ptype = "linear_gprior"

    def __post_init__(self):
        LinearPrior.__post_init__(self)
        GaussianPrior.__post_init__(self)


@dataclass
class LinearUniformPrior(LinearPrior, UniformPrior):
    ptype = "linear_uprior"

    def __post_init__(self):
        LinearPrior.__post_init__(self)
        UniformPrior.__post_init__(self)


@dataclass
class SplineGaussianPrior(SplinePrior, GaussianPrior):
    ptype = "linear_gprior"

    def __post_init__(self):
        SplinePrior.__post_init__(self)
        GaussianPrior.__post_init__(self)


@dataclass
class SplineUniformPrior(SplinePrior, UniformPrior):
    ptype = "linear_uprior"

    def __post_init__(self):
        SplinePrior.__post_init__(self)
        UniformPrior.__post_init__(self)
