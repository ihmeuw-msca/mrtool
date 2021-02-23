# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~

    Covariates model for `mrtool`.
"""
import operator
from typing import Callable, Iterable, List, Union, Dict
from collections import defaultdict

import numpy as np
from mrtool.core import utils
from mrtool.core.data import MRData
from numpy import ndarray
from mrtool.core.prior import (GaussianPrior, LinearPrior, Prior, SplinePrior,
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
                 spline: Union[XSpline, SplineSpecs] = None,
                 priors: Iterable[Prior] = None):

        self.alt_cov = Covariate(alt_cov)
        self.ref_cov = Covariate(ref_cov)
        self.covs = self.alt_cov.name + self.ref_cov.name

        self.spline = spline

        self.priors = [] if priors is None else list(priors)
        self.sorted_priors = defaultdict(list)
        self.sort_priors()
        self.activate_spline_priors()

    @property
    def use_spline(self) -> bool:
        return self.spline is not None

    @property
    def size(self) -> int:
        if self.use_spline:
            return self.spline.num_spline_bases
        return 1

    def sort_priors(self) -> Dict[str, List[Prior]]:
        """sort priors.
        """
        for prior in self.priors:
            self.sorted_priors[prior.ptype].append(prior)

        if ((len(self.sorted_priors["gprior"]) >= 2 or
             len(self.sorted_priors["uprior"]) >= 2)):
            raise ValueError("Cannot have multiple gprior or uprior.")
        if ((len(self.sorted_priors["linear_gprior"]) >= 1 or
             len(self.sorted_priors["linear_uprior"]) >= 1) and
                (not self.use_spline)):
            raise ValueError("Cannot have linear priors for uni-variate.")

    def activate_spline_priors(self):
        if isinstance(self.spline, XSpline):
            for prior in (self.sort_priors["linear_uprior"] +
                          self.sort_priors["linear_gprior"]):
                if isinstance(prior, SplinePrior):
                    prior.attach_spline(self.spline)

    def attach_data(self, data: MRData):
        """Attach data.
        """
        if isinstance(self.spline, SplineSpecs):
            self.spline = self.spline.create_spline(data[self.covs])
        self.activate_spline_priors()

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
        gprior = self.sorted_priors["gprior"]
        if not gprior:
            gvec = np.repeat([[0.0], [np.inf]], self.size, axis=1)
        else:
            gvec = np.vstack([gprior[0].mean, gprior[0].sd])
        return gvec

    def get_uvec(self) -> np.ndarray:
        uprior = self.sorted_priors["uprior"]
        if not uprior:
            uvec = np.repeat([[-np.inf], [np.inf]], self.size, axis=1)
        else:
            uvec = np.vstack([uprior[0].lb, uprior[0].ub])
        return uvec

    def get_linear_uvec(self) -> np.ndarray:
        linear_uprior = self.sorted_priors["linear_uprior"]
        if not linear_uprior:
            uvec = np.empty((2, 0))
        else:
            uvec = np.hstack([
                np.vstack([prior.lb, prior.ub])
                for prior in linear_uprior
            ])
        return uvec

    def get_linear_gvec(self) -> np.ndarray:
        linear_gprior = self.sorted_priors["linear_gprior"]
        if not linear_gprior:
            gvec = np.empty((2, 0))
        else:
            gvec = np.hstack([
                np.vstack([prior.mean, prior.sd])
                for prior in linear_gprior
            ])
        return gvec

    def get_linear_umat(self, data: MRData = None) -> np.ndarray:
        self.attach_data(data)
        linear_uprior = self.sorted_priors["linear_uprior"]
        if not linear_uprior:
            umat = np.empty((0, self.size))
        else:
            umat = np.vstack([
                prior.mat for prior in linear_uprior
            ])
        return umat

    def get_linear_gmat(self, data: MRData = None) -> np.ndarray:
        self.attach_data(data)
        linear_gprior = self.sorted_priors["linear_gprior"]
        if not linear_gprior:
            gmat = np.empty((0, self.size))
        else:
            gmat = np.vstack([
                prior.mat for prior in linear_gprior
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
