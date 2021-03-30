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
from mrtool.core.prior import Prior, SplinePrior, default_prior_params
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
            for prior in (self.sorted_priors["linear_uprior"] +
                          self.sorted_priors["linear_gprior"]):
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

    def get_priors(self, ptype: str):
        arr = self.get_prior_array(ptype)
        mat, vec = arr[0], arr[1]
        if mat is not None:
            if mat.shape[1] != self.size:
                raise ValueError("Linear prior size not match.")
            result = (mat, vec)
        else:
            if vec.shape[1] != self.size:
                raise ValueError("Prior size not match")
            result = vec
        return result

    def get_prior_array(self, ptype: str):
        priors = self.sorted_priors[ptype]
        arr = [None, None]
        if not priors:
            if "linear" in ptype:
                arr[0] = np.empty((0, self.size))
                arr[1] = np.empty((2, 0))
            else:
                arr[1] = np.repeat(default_prior_params[ptype], self.size, axis=1)
        else:
            arr[1] = np.hstack([prior.info for prior in priors])
            if "linear" in ptype:
                arr[0] = np.vstack([prior.mat for prior in priors])
            else:
                if arr[1].shape[1] == 1:
                    arr[1] = np.repeat(arr[1], self.size, axis=1)

        return arr

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(alt_cov={self.alt_cov.name}, "
                f"ref_cov={self.ref_cov.name}, "
                f"use_spline={self.use_spline})")


class LinearCovModel(CovModel):
    """Linear Covariates Model.
    """

    def get_fun(self, data: MRData) -> Callable:
        alt_mat = self.alt_cov.get_design_mat(data, self.spline)
        ref_mat = self.ref_cov.get_design_mat(data, self.spline)
        return utils.mat_to_fun(alt_mat, ref_mat=ref_mat)


class LogCovModel(CovModel):
    """Log Covariates Model.
    """
    # TODO: add the prior for positive constraint of value

    def get_fun(self, data):
        self.attach_data(data)
        alt_mat = self.alt_cov.get_design_mat(data, self.spline)
        ref_mat = self.ref_cov.get_design_mat(data, self.spline)
        add_one = not (self.use_spline and self.spline.include_first_basis)
        return utils.mat_to_log_fun(alt_mat, ref_mat=ref_mat, add_one=add_one)
