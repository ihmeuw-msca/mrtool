# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~

    Covariates model for `mrtool`.
"""
import operator
from typing import Union, Iterable, Callable
import numpy as np
import xspline
from numpy import ndarray
from xspline import XSpline
from regmod.prior import (Prior, GaussianPrior, UniformPrior,
                          LinearGaussianPrior, LinearUniformPrior,
                          SplineGaussianPrior, SplineUniformPrior)
from . import utils
from mrtool.core.data import MRData


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
                       spline: XSpline = None,
                       use_spline_intercept: bool = False) -> ndarray:
        mat = self.get_mat(data)
        design_mat = utils.avg_integral(mat, spline, use_spline_intercept)
        return design_mat

    def __repr__(self) -> str:
        return f"Covariate(name={self.name})"


class CovModel:
    """Covariates model.
    """

    def __init__(self,
                 alt_cov,
                 name=None,
                 ref_cov=None,
                 use_re=False,
                 use_re_mid_point=False,
                 use_spline=False,
                 use_spline_intercept=False,
                 spline_knots_type='frequency',
                 spline_knots=np.linspace(0.0, 1.0, 4),
                 spline_degree=3,
                 spline_l_linear=False,
                 spline_r_linear=False,
                 priors: Iterable[Prior] = None):
        """Constructor of the covariate model.

        Args:
            alt_cov(str | list{str}):
                Main covariate name, when it is a list consists of two
                covariates names, use the average integrand between defined by
                the two covariates.
            ref_cov(str | list{str} | None, optional):
                Reference covariate name, will be interpreted differently in the
                sub-classes.
            name(str | None, optional):
                Model name for easy access.
            use_re(bool, optional):
                If use the random effects.
            use_re_mid_point(bool, optional):
                If use the midpoint for the random effects.
            use_spline(bool, optional):
                If use splines.
            use_spline_intercept(bool, optional):
                If `True`, use full set of the spline bases, shouldn't include extra `intercept` in this case.
            spline_knots_type(str, optional):
                The method of how to place the knots, `'frequency'` place the
                knots according to the data quantile and `'domain'` place the
                knots according to the domain of the data.
            spline_knots(np.ndarray, optional):
                A numpy array between 0 and 1 contains the relative position of
                the knots placement, with respect to either frequency or domain.
            spline_degree(int, optional):
                The degree of the spline.
            spline_l_linear(bool, optional):
                If use left linear tail.
            spline_r_linear(bool, optional):
                If use right linear tail.
        """
        self.alt_cov = Covariate(alt_cov)
        self.ref_cov = Covariate(ref_cov)
        self.covs = self.alt_cov.name + self.ref_cov.name

        self.name = name
        self.use_re = use_re
        self.use_re_mid_point = use_re_mid_point
        self.use_spline = use_spline
        self.use_spline_intercept = use_spline_intercept and self.use_spline

        self.spline = None
        self.spline_knots_type = spline_knots_type
        self.spline_knots_template = np.asarray(spline_knots)
        self.spline_knots = None
        self.spline_degree = spline_degree
        self.spline_l_linear = spline_l_linear
        self.spline_r_linear = spline_r_linear

        self.priors = [] if priors is None else list(priors)
        self.uprior = None
        self.gprior = None
        self.linear_gpriors = []
        self.linear_upriors = []

        self._check_inputs()
        self._process_inputs()
        self._process_priors()

    def _check_inputs(self):
        """Check the attributes.
        """
        assert isinstance(self.name, str) or self.name is None
        assert isinstance(self.use_re, bool)
        assert isinstance(self.use_spline, bool)

        # spline specific
        assert self.spline is None or isinstance(self.spline, xspline.XSpline)
        assert self.spline_knots_type in ['frequency', 'domain']
        assert isinstance(self.spline_knots_template, np.ndarray)
        assert np.min(self.spline_knots_template) >= 0.0
        assert np.max(self.spline_knots_template) <= 1.0
        assert isinstance(self.spline_degree, int)
        assert self.spline_degree >= 0
        assert isinstance(self.spline_l_linear, bool)
        assert isinstance(self.spline_r_linear, bool)

    def _process_inputs(self):
        """Process attributes.
        """
        # model name
        if self.name is None:
            if len(self.alt_cov) == 1:
                self.name = self.alt_cov[0]
            else:
                self.name = 'cov' + '{:0>3}'.format(np.random.randint(1000))

        # spline knots
        self.spline_knots_template = np.hstack([self.spline_knots_template, [0.0, 1.0]])
        self.spline_knots_template = np.unique(self.spline_knots_template)

    # TODO: in the spline PR change the size of this
    @property
    def size(self) -> int:
        return 1

    def _process_priors(self):
        """Process priors.
        """
        for prior in self.priors:
            if isinstance(prior, (SplineGaussianPrior, LinearGaussianPrior)):
                self.linear_gpriors.append(prior)
            elif isinstance(prior, (SplineUniformPrior, LinearUniformPrior)):
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
        if self.use_spline:
            self.spline = self.create_spline(data, spline_knots=self.spline_knots)
            self.spline_knots = self.spline.knots

    # TODO: remove this function
    def has_data(self):
        """Return ``True`` if there is one data object attached.
        """
        if self.use_spline:
            return self.spline is not None
        else:
            return True

    # TODO: change this function in spline PR using spline_specs
    def create_spline(self, data: MRData, spline_knots: np.ndarray = None) -> xspline.XSpline:
        """Create spline given current spline parameters.
        Args:
            data(mrtool.MRData):
                The data frame used for storing the data
            spline_knots(np.ndarray, optional):
                Spline knots, if ``None`` determined by frequency or domain.
        Returns:
            xspline.XSpline: The spline object.
        """
        # extract covariate
        alt_cov = self.alt_cov.get_mat(data)
        ref_cov = self.ref_cov.get_mat(data)

        cov_all = np.hstack((alt_cov.ravel(), ref_cov.ravel()))
        cov = np.array([min(cov_all), max(cov_all)])
        if alt_cov.size != 0:
            cov = np.hstack((cov, alt_cov.mean(axis=1)))
        if ref_cov.size != 0:
            cov = np.hstack((cov, ref_cov.mean(axis=1)))
        cov = np.unique(cov)

        if spline_knots is None:
            if self.spline_knots_type == 'frequency':
                spline_knots = np.quantile(cov, self.spline_knots_template)
            else:
                spline_knots = cov.min() + self.spline_knots_template*(cov.max() - cov.min())

        spline = xspline.XSpline(spline_knots,
                                 self.spline_degree,
                                 l_linear=self.spline_l_linear,
                                 r_linear=self.spline_r_linear)

        return spline

    def get_mat(self, data):
        """Create design matrix.
        """
        alt_mat = self.alt_cov.get_design_mat(data,
                                              self.spline,
                                              self.use_spline_intercept)
        ref_mat = self.ref_cov.get_design_mat(data,
                                              self.spline,
                                              self.use_spline_intercept)
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
        if not self.linear_upriors:
            umat = np.empty((0, self.size))
        else:
            umat = np.vstack([
                prior.mat for prior in self.linear_upriors
            ])
        return umat

    def get_linear_gmat(self, data: MRData = None) -> np.ndarray:
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
        alt_mat = self.alt_cov.get_design_mat(data,
                                              self.spline,
                                              self.use_spline_intercept)
        ref_mat = self.ref_cov.get_design_mat(data,
                                              self.spline,
                                              self.use_spline_intercept)
        return utils.mat_to_fun(alt_mat, ref_mat=ref_mat)


class LogCovModel(CovModel):
    """Log Covariates Model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_fun(self, data):
        """Create design functions for the fixed effects.

        Args:
            data(mrtool.MRData):
                The data frame used for storing the data

        Returns:
            tuple{function, function}:
                Design functions for fixed effects.
        """
        alt_mat = self.alt_cov.get_design_mat(data,
                                              self.spline,
                                              self.use_spline_intercept)
        ref_mat = self.ref_cov.get_design_mat(data,
                                              self.spline,
                                              self.use_spline_intercept)
        add_one = not (self.use_spline and self.use_spline_intercept)
        return utils.mat_to_log_fun(alt_mat, ref_mat=ref_mat, add_one=add_one)

    # TODO: add this in the init of the priors
    # def create_constraint_mat(self, threshold=1e-6):
    #     """Create constraint matrix.
    #     Overwrite the super class, adding non-negative constraints.
    #     """
    #     c_mat, c_val = super().create_constraint_mat()
    #     shift = 0.0 if self.use_spline_intercept else 1.0
    #     index = 0 if self.use_spline_intercept else 1
    #     tmp_val = np.array([[-shift + threshold], [np.inf]])
    #     if self.use_spline:
    #         points = np.linspace(self.spline.knots[0],
    #                              self.spline.knots[-1],
    #                              self.prior_spline_num_constraint_points)
    #         c_mat = np.vstack((c_mat, self.spline.design_mat(points)[:, index:]))
    #         c_val = np.hstack((c_val, np.repeat(tmp_val, points.size, axis=1)))
    #     return c_mat, c_val

    # @property
    # def num_constraints(self):
    #     num_c = super().num_constraints
    #     if self.use_spline:
    #         num_c += self.prior_spline_num_constraint_points
    #     return num_c

    # @property
    # def num_z_vars(self):
    #     return int(self.use_re)
