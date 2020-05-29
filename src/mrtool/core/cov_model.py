# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~

    Covariates model for `mrtool`.
"""
from typing import Tuple
import numpy as np
import xspline
from . import utils
from .data import MRData


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
                 spline_knots_type='frequency',
                 spline_knots=np.linspace(0.0, 1.0, 4),
                 spline_degree=3,
                 spline_l_linear=False,
                 spline_r_linear=False,
                 prior_spline_monotonicity=None,
                 prior_spline_convexity=None,
                 prior_spline_num_constraint_points=20,
                 prior_spline_maxder_gaussian=None,
                 prior_spline_maxder_uniform=None,
                 prior_beta_gaussian=None,
                 prior_beta_uniform=None,
                 prior_beta_laplace=None,
                 prior_gamma_gaussian=None,
                 prior_gamma_uniform=None,
                 prior_gamma_laplace=None):
        """Constructor of the covariate model.

        Args:
            alt_cov (str | list{str}):
                Main covariate name, when it is a list consists of two
                covariates names, use the average integrand between defined by
                the two covariates.
            ref_cov (str | list{str} | None, optional):
                Reference covariate name, will be interpreted differently in the
                sub-classes.
            name (str | None, optional):
                Model name for easy access.
            use_re (bool, optional):
                If use the random effects.
            use_re_mid_point (bool, optional):
                If use the midpoint for the random effects.
            use_spline(bool, optional):
                If use splines.
            spline_knots_type (str, optional):
                The method of how to place the knots, `'frequency'` place the
                knots according to the data quantile and `'domain'` place the
                knots according to the domain of the data.
            spline_knots (np.ndarray, optional):
                A numpy array between 0 and 1 contains the relative position of
                the knots placement, with respect to either frequency or domain.
            spline_degree (int, optional):
                The degree of the spline.
            spline_l_linear (bool, optional):
                If use left linear tail.
            spline_r_linear (bool, optional):
                If use right linear tail.
            prior_spline_monotonicity (str | None, optional):
                Spline shape prior, `'increasing'` indicates spline is
                increasing, `'decreasing'` indicates spline is decreasing.
            prior_spline_convexity (str | None, optional):
                Spline shape prior, `'convex'` indicate if spline is convex and
                `'concave'` indicate spline is concave.
            prior_spline_num_constraint_points (int, optional):
                Number of constraint points used in the the shape constraints
                of the spline.
            prior_spline_maxder_gaussian (numpy.ndarray, optional):
                Gaussian prior on the highest derivative of the spline.
                When it is a one dimensional array, the first element will be
                the mean for all derivative and second element will be the sd.
                When it is a two dimensional array, the first row will be the
                mean and the second row will be the sd, the number of columns
                should match the number of the intervals defined by the spline
                knots.
            prior_spline_maxder_uniform (numpy.ndarray, optional)
                Uniform prior on the highest derivative of the spline.
            prior_beta_gaussian (numpy.ndarray, optional):
                Direct Gaussian prior for beta. It can be one dimensional or
                two dimensional array like `prior_spline_maxder_gaussian`.
            prior_beta_uniform (numpy.ndarray, optional):
                Direct uniform prior for beta.
            prior_beta_laplace (numpy.ndarray, optional):
                Direct Laplace prior for beta.
            prior_gamma_gaussian (numpy.ndarray, optional):
                Direct Gaussian prior for gamma.
            prior_gamma_uniform (numpy.ndarray, optional):
                Direct uniform prior for gamma.
            prior_gamma_laplace (numpy.ndarray, optional):
                Direct Laplace prior for gamma.
        """
        self.covs = []
        self.alt_cov = utils.input_cols(alt_cov, append_to=self.covs)
        self.ref_cov = utils.input_cols(ref_cov, append_to=self.covs)

        self.name = name
        self.use_re = use_re
        self.use_re_mid_point = use_re_mid_point
        self.use_spline = use_spline

        self.spline = None
        self.spline_knots_type = spline_knots_type
        self.spline_knots_template = spline_knots
        self.spline_knots = None
        self.spline_degree = spline_degree
        self.spline_l_linear = spline_l_linear
        self.spline_r_linear = spline_r_linear

        self.prior_spline_monotonicity = prior_spline_monotonicity
        self.prior_spline_convexity = prior_spline_convexity
        self.prior_spline_num_constraint_points = prior_spline_num_constraint_points
        self.prior_spline_maxder_gaussian = prior_spline_maxder_gaussian
        self.prior_spline_maxder_uniform = prior_spline_maxder_uniform
        self.prior_beta_gaussian = prior_beta_gaussian
        self.prior_beta_uniform = prior_beta_uniform
        self.prior_beta_laplace = prior_beta_laplace
        self.prior_gamma_gaussian = prior_gamma_gaussian
        self.prior_gamma_uniform = prior_gamma_uniform
        self.prior_gamma_laplace = prior_gamma_laplace

        self._check_inputs()
        self._process_inputs()
        if not self.use_spline:
            self._process_priors()

    def _check_inputs(self):
        """Check the attributes.
        """
        assert utils.is_cols(self.alt_cov)
        assert utils.is_cols(self.ref_cov)
        assert isinstance(self.name, str) or self.name is None
        if isinstance(self.alt_cov, list):
            assert len(self.alt_cov) <= 2
        if isinstance(self.ref_cov, list):
            assert len(self.ref_cov) <= 2
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

        # priors
        assert (self.prior_spline_monotonicity in ['increasing', 'decreasing'] or
                self.prior_spline_monotonicity is None)
        assert (self.prior_spline_convexity in ['convex', 'concave'] or
                self.prior_spline_convexity is None)
        assert isinstance(self.prior_spline_num_constraint_points, int)
        assert self.prior_spline_num_constraint_points > 0
        assert utils.is_gaussian_prior(self.prior_spline_maxder_gaussian)
        assert utils.is_gaussian_prior(self.prior_beta_gaussian)
        assert utils.is_gaussian_prior(self.prior_gamma_gaussian)
        assert utils.is_uniform_prior(self.prior_spline_maxder_uniform)
        assert utils.is_uniform_prior(self.prior_beta_uniform)
        assert utils.is_uniform_prior(self.prior_gamma_uniform)
        assert utils.is_laplace_prior(self.prior_beta_laplace)
        assert utils.is_laplace_prior(self.prior_gamma_laplace)

    def _process_inputs(self):
        """Process attributes.
        """
        # covariates names
        if not isinstance(self.alt_cov, list):
            self.alt_cov = [self.alt_cov]
        if not isinstance(self.ref_cov, list):
            self.ref_cov = [self.ref_cov]
        # model name
        if self.name is None:
            if len(self.alt_cov) == 1:
                self.name = self.alt_cov[0]
            else:
                self.name = 'cov' + '{:0>3}'.format(np.random.randint(1000))

        # spline knots
        self.spline_knots_template = np.hstack([self.spline_knots_template, [0.0, 1.0]])
        self.spline_knots_template = np.unique(self.spline_knots_template)

    def _process_priors(self):
        """Process priors.
        """
        # prior information
        if self.spline is not None:
            self.prior_spline_maxder_gaussian = utils.input_gaussian_prior(
                self.prior_spline_maxder_gaussian, self.spline_knots.size - 1
            )
            self.prior_spline_maxder_uniform = utils.input_uniform_prior(
                self.prior_spline_maxder_uniform, self.spline_knots.size - 1
            )
        else:
            self.prior_spline_maxder_gaussian = None
            self.prior_spline_maxder_uniform = None
        self.prior_beta_gaussian = utils.input_gaussian_prior(
            self.prior_beta_gaussian, self.num_x_vars
        )
        self.prior_beta_uniform = utils.input_uniform_prior(
            self.prior_beta_uniform, self.num_x_vars
        )
        self.prior_beta_laplace = utils.input_laplace_prior(
            self.prior_beta_laplace, self.num_x_vars
        )
        self.prior_gamma_gaussian = utils.input_gaussian_prior(
            self.prior_gamma_gaussian, self.num_z_vars
        )
        self.prior_gamma_uniform = utils.input_uniform_prior(
            self.prior_gamma_uniform, self.num_z_vars
        )
        self.prior_gamma_uniform = np.maximum(0.0, self.prior_gamma_uniform)
        self.prior_gamma_laplace = utils.input_laplace_prior(
            self.prior_gamma_laplace, self.num_z_vars
        )

    def attach_data(self, data: MRData):
        """Attach data.
        """
        if self.use_spline:
            self.spline = self.create_spline(data)
            self.spline_knots = self.spline.knots
            self._process_priors()

    def create_spline(self, data: MRData) -> xspline.XSpline:
        """Create spline given current spline parameters.
        Args:
            data (mrtool.MRData):
                The data frame used for storing the data
        Returns:
            xspline.XSpline: The spline object.
        """
        # extract covariate
        alt_cov = data.get_covs(self.alt_cov)
        ref_cov = data.get_covs(self.ref_cov)

        cov_all = np.hstack((alt_cov.ravel(), ref_cov.ravel()))
        cov = np.array([min(cov_all), max(cov_all)])
        if alt_cov.size != 0:
            cov = np.hstack((cov, alt_cov.mean(axis=1)))
        if ref_cov.size != 0:
            cov = np.hstack((cov, ref_cov.mean(axis=1)))

        if self.spline_knots_type == 'frequency':
            spline_knots = np.quantile(cov, self.spline_knots_template)
        else:
            spline_knots = min(cov) + self.spline_knots_template*(max(cov) - min(cov))

        spline = xspline.XSpline(spline_knots,
                                 self.spline_degree,
                                 l_linear=self.spline_l_linear,
                                 r_linear=self.spline_r_linear)

        return spline

    def create_design_mat(self, data):
        """Create design matrix.
        Args:
            data (mrtool.MRData):
                The data frame used for storing the data
        Returns:
            tuple{numpy.ndarray, numpy.ndarray}:
                Return the design matrix for linear cov or spline.
        """
        alt_cov = data.get_covs(self.alt_cov)
        ref_cov = data.get_covs(self.ref_cov)

        alt_mat = utils.avg_integral(alt_cov, spline=self.spline)
        ref_mat = utils.avg_integral(ref_cov, spline=self.spline)

        return alt_mat, ref_mat

    def create_x_fun(self, data):
        raise NotImplementedError("Cannot use create_x_fun directly in CovModel class.")

    def create_z_mat(self, data):
        raise NotImplementedError("Cannot use create_z_mat directly in CovModel class.")

    def create_constraint_mat(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create constraint matrix.
        Returns:
            tuple{numpy.ndarray, numpy.ndarray}:
                Return linear constraints matrix and its uniform prior.
        """
        # initialize the matrix and the value
        c_mat = np.array([]).reshape(0, self.num_x_vars)
        c_val = np.array([]).reshape(2, 0)
        if not self.use_spline:
            return c_mat, c_val

        points = np.linspace(self.spline.knots[0],
                             self.spline.knots[-1],
                             self.prior_spline_num_constraint_points)
        tmp_val = np.array([[-np.inf], [0.0]])

        # spline monotonicity constraints
        if self.prior_spline_monotonicity is not None and self.use_spline:
            sign = 1.0 if self.prior_spline_monotonicity is 'decreasing' else -1.0
            c_mat = np.vstack((c_mat, sign*self.spline.design_dmat(points, 1)[:, 1:]))
            c_val = np.hstack((c_val, np.repeat(tmp_val, points.size, axis=1)))

        # spline convexity constraints
        if self.prior_spline_convexity is not None and self.use_spline:
            sign = 1.0 if self.prior_spline_convexity is 'concave' else -1.0
            c_mat = np.vstack((c_mat, sign*self.spline.design_dmat(points, 2)[:, 1:]))
            c_val = np.hstack((c_val, np.repeat(tmp_val, points.size, axis=1)))

        # spline maximum derivative constraints
        if not np.isinf(self.prior_spline_maxder_uniform).all():
            c_mat = np.vstack((c_mat, self.spline.last_dmat()[:, 1:]))
            c_val = np.hstack((c_val, self.prior_spline_maxder_uniform))

        return c_mat, c_val

    def create_regularization_mat(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create constraint matrix.
        Returns:
            tuple{numpy.ndarray, numpy.ndarray}:
                Return linear regularization matrix and its Gaussian prior.
        """
        r_mat = np.array([]).reshape(0, self.num_x_vars)
        r_val = np.array([]).reshape(2, 0)
        if not self.use_spline:
            return r_mat, r_val

        # spline maximum derivative constraints
        if not np.isinf(self.prior_spline_maxder_gaussian[1]).all() and self.use_spline:
            r_mat = np.vstack((r_mat, self.spline.last_dmat()[:, 1:]))
            r_val = np.hstack((r_val, self.prior_spline_maxder_gaussian))

        return r_mat, r_val

    @property
    def num_x_vars(self):
        if self.use_spline:
            assert self.spline is not None, "Attach data first to create spline."
            n = self.spline.num_spline_bases - 1
        else:
            n = 1
        return n

    @property
    def num_z_vars(self):
        if self.use_re:
            if self.use_re_mid_point:
                return 1
            else:
                return self.num_x_vars
        else:
            return 0

    @property
    def num_constraints(self):
        if not self.use_spline:
            return 0
        else:
            num_c = self.prior_spline_num_constraint_points*(
                    (self.prior_spline_monotonicity is not None) +
                    (self.prior_spline_convexity is not None)
            )
            if not np.isinf(self.prior_spline_maxder_uniform).all():
                num_c += self.prior_spline_maxder_uniform.shape[1]

            return num_c

    @property
    def num_regularizations(self):
        if not self.use_spline:
            return 0
        else:
            num_r = 0
            if not np.isinf(self.prior_spline_maxder_gaussian[1]).all():
                num_r += self.prior_spline_maxder_gaussian.shape[1]

            return num_r


class LinearCovModel(CovModel):
    """Linear Covariates Model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_x_fun(self, data: MRData):
        """Create design function for the fixed effects.
        """
        alt_mat, ref_mat = self.create_design_mat(data)
        return utils.mat_to_fun(alt_mat, ref_mat=ref_mat)

    def create_z_mat(self, data):
        """Create design matrix for the random effects.

        Args:
            data (mrtool.MRData):
                The data frame used for storing the data

        Returns:
            numpy.ndarray:
                Design matrix for random effects.
        """
        if not self.use_re:
            return np.array([]).reshape(data.num_obs, 0)

        if self.use_re_mid_point:
            alt_mat = utils.avg_integral(data.get_covs(self.alt_cov))
            ref_mat = utils.avg_integral(data.get_covs(self.ref_cov))
        else:
            alt_mat, ref_mat = self.create_design_mat(data)

        if ref_mat.size == 0:
            return alt_mat
        else:
            return alt_mat - ref_mat


class LogCovModel(CovModel):
    """Log Covariates Model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_x_fun(self, data):
        """Create design functions for the fixed effects.

        Args:
            data (mrtool.MRData):
                The data frame used for storing the data

        Returns:
            tuple{function, function}:
                Design functions for fixed effects.
        """
        alt_mat, ref_mat = self.create_design_mat(data)
        return utils.mat_to_log_fun(alt_mat, ref_mat=ref_mat)

    def create_z_mat(self, data):
        """Create design matrix for the random effects.

        Args:
            data (mrtool.MRData):
                The data frame used for storing the data

        Returns:
            numpy.ndarray:
                Design matrix for random effects.
        """
        if not self.use_re:
            return np.array([]).reshape(data.num_obs, 0)

        alt_mat = utils.avg_integral(data.get_covs(self.alt_cov))
        ref_mat = utils.avg_integral(data.get_covs(self.ref_cov))

        if ref_mat.size == 0:
            return alt_mat
        else:
            return alt_mat - ref_mat

    def create_constraint_mat(self):
        """Create constraint matrix.
        Overwrite the super class, adding non-negative constraints.
        """
        c_mat, c_val = super().create_constraint_mat()
        tmp_val = np.array([[-1.0], [np.inf]])

        if self.use_spline:
            points = np.linspace(self.spline.knots[0],
                                 self.spline.knots[-1],
                                 self.prior_spline_num_constraint_points)
            c_mat = np.vstack((c_mat, self.spline.design_mat(points)[:, 1:]))
            c_val = np.hstack((c_val, np.repeat(tmp_val, points.size, axis=1)))
        return c_mat, c_val

    @property
    def num_constraints(self):
        num_c = super().num_constraints
        if self.use_spline:
            num_c += self.prior_spline_num_constraint_points
        return num_c

    @property
    def num_z_vars(self):
        return int(self.use_re)
