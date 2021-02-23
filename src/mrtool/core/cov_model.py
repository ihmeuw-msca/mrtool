# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~

    Covariates model for `mrtool`.
"""
import operator
from typing import Callable, Iterable, Union

import numpy as np
from mrtool.core import utils
from mrtool.core.data import MRData
from numpy import ndarray
from regmod.prior import (GaussianPrior, LinearPrior, Prior, SplinePrior,
                          UniformPrior)
from regmod.utils import SplineSpecs
from xspline import XSpline


class Covariate:
    """
    Covariate class, main responsibility is to create design mat.
    """

    def __init__(self, name: Union[None, str, Iterable[str]] = None):
        self.name = name

    name = property(operator.attrgetter("_name"))

    @name.setter
    def name(self, covs: Union[None, str, Iterable[str]]):
        if not (covs is None or
                isinstance(covs, str) or
                all([isinstance(cov, str) for cov in covs])):
            raise TypeError("Covaraite name can only be None or string or list of strings.")

        covs = [] if covs is None else [covs] if isinstance(covs, str) else list(covs)

        if len(covs) not in [0, 1, 2]:
            raise ValueError("Covariate can only contain zero, one or two name(s).")

        self._name = covs

    @property
    def is_empty(self) -> bool:
        return len(self.name) == 0

    def get_mat(self, data: MRData) -> ndarray:
        return data[self.name]

    def get_design_mat(self,
                       data: MRData,
                       spline: XSpline = None) -> ndarray:
        mat = self.get_mat(data)
        design_mat = utils.avg_integral(mat, spline)
        return design_mat

    def __repr__(self) -> str:
        return f"Covariate(name={self.name})"


class CovModel:
    """Covariates model.
    """

    def __init__(self,
                 alt_cov: Union[str, Iterable[str]],
                 ref_cov: Union[str, Iterable[str]] = None,
                 spline: XSpline = None,
                 spline_specs: SplineSpecs = None,
                 priors: Iterable[Prior] = None):

        self.alt_cov = Covariate(alt_cov)
        self.ref_cov = Covariate(ref_cov)
        self.covs = self.alt_cov.name + self.ref_cov.name

        self.spline = spline
        self.spline_specs = spline_specs

        self.priors = [] if priors is None else list(priors)
        self.uprior = None
        self.gprior = None
        self.linear_gpriors = []
        self.linear_upriors = []

        self._process_priors()

    @property
    def use_spline(self) -> bool:
        return (self.spline is not None) or (self.spline_specs is not None)

    @property
    def size(self) -> int:
        if self.use_spline:
            if self.spline is not None:
                return self.spline.num_spline_bases
            return self.spline_specs.num_spline_bases
        return 1

    def _process_priors(self):
        """Process priors.
        """
        for prior in self.priors:
            if isinstance(prior, LinearPrior) and isinstance(prior, GaussianPrior):
                assert self.use_spline, "Cannot use linear prior on uni-var."
                self.linear_gpriors.append(prior)
            elif isinstance(prior, LinearPrior) and isinstance(prior, UniformPrior):
                assert self.use_spline, "Cannot use linear prior on uni-var."
                self.linear_upriors.append(prior)
            elif isinstance(prior, GaussianPrior):
                if self.gprior is not None and self.gprior != prior:
                    raise ValueError("Can only provide one Gaussian prior.")
                self.gprior = prior
                assert self.gprior.size == self.size, \
                    "Gaussian prior size not match."
            elif isinstance(prior, UniformPrior):
                if self.uprior is not None and self.uprior != prior:
                    raise ValueError("Can only provide one Uniform prior.")
                self.uprior = prior
                assert self.uprior.size == self.size, \
                    "Uniform prior size not match."
            else:
                raise ValueError("Unknown prior type.")

    def attach_data(self, data: MRData):
        """Attach data.
        """
        if self.use_spline and self.spline is None:
            self.spline = self.spline_specs.create_spline(data[self.covs])
            for prior in self.linear_upriors + self.linear_gpriors:
                if isinstance(prior, SplinePrior):
                    prior.attach_spline(self.spline)

    def get_mat(self, data):
        """Create design matrix.
        """
        self.attach_data(data)
        alt_mat = self.alt_cov.get_design_mat(data, self.spline)
        ref_mat = self.ref_cov.get_design_mat(data, self.spline)
        mat = alt_mat if ref_mat.size == 0 else alt_mat - ref_mat
        return mat

    def get_fun(self, data):
        """
        Create design function
        """
        raise NotImplementedError("Do not directly use CovModel class.")

    def get_gvec(self) -> np.ndarray:
        if self.gprior is None:
            gvec = np.repeat([[0.0], [np.inf]], self.size, axis=1)
        else:
            gvec = np.vstack([self.gprior.mean, self.gprior.sd])
        return gvec

    def get_uvec(self) -> np.ndarray:
        if self.uprior is None:
            uvec = np.repeat([[-np.inf], [np.inf]], self.size, axis=1)
        else:
            uvec = np.vstack([self.uprior.lb, self.uprior.ub])
        return uvec

    def get_linear_uvec(self) -> np.ndarray:
        if not self.linear_upriors:
            uvec = np.empty((2, 0))
        else:
            uvec = np.hstack([
                np.vstack([prior.lb, prior.ub])
                for prior in self.linear_upriors
            ])
        return uvec

    def get_linear_gvec(self) -> np.ndarray:
        if not self.linear_gpriors:
            gvec = np.empty((2, 0))
        else:
            gvec = np.hstack([
                np.vstack([prior.mean, prior.sd])
                for prior in self.linear_gpriors
            ])
        return gvec

    def get_linear_umat(self, data: MRData = None) -> np.ndarray:
        self.attach_data(data)
        if not self.linear_upriors:
            umat = np.empty((0, self.size))
        else:
            umat = np.vstack([
                prior.mat for prior in self.linear_upriors
            ])
        return umat

    def get_linear_gmat(self, data: MRData = None) -> np.ndarray:
        self.attach_data(data)
        if not self.linear_gpriors:
            gmat = np.empty((0, self.size))
        else:
            gmat = np.vstack([
                prior.mat for prior in self.linear_gpriors
            ])
        return gmat


class LinearCovModel(CovModel):
    """Linear Covariates Model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_fun(self, data: MRData) -> Callable:
        alt_mat = self.alt_cov.get_design_mat(data, self.spline)
        ref_mat = self.ref_cov.get_design_mat(data, self.spline)
        return utils.mat_to_fun(alt_mat, ref_mat=ref_mat)


class LogCovModel(CovModel):
    """Log Covariates Model.
    """

    def __init__(self, *args, **kwargs):
        # TODO: add the prior for positive constraint of value
        super().__init__(*args, **kwargs)

    def get_fun(self, data):
        self.attach_data(data)
        alt_mat = self.alt_cov.get_design_mat(data, self.spline)
        ref_mat = self.ref_cov.get_design_mat(data, self.spline)
        add_one = not (self.use_spline and self.spline.include_first_basis)
        return utils.mat_to_log_fun(alt_mat, ref_mat=ref_mat, add_one=add_one)
