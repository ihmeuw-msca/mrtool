# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~

    Covariates model for `mrtool`.
"""
import numpy as np
import xspline
from . import utils


class CovModel:
    """Covariates model.
    """
    def __init__(self,
                 alt_cov,
                 ref_cov=None,
                 use_re=False,
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
                 prior_gamma_gaussian=None,
                 prior_gamma_uniform=None):
        """Constructor of the covariate model.

        Args:
            alt_cov (str | list{str}):
                Main covariate name, when it is a list consists of two
                covariates names, use the average integrand between defined by
                the two covariates.
            ref_cov (str | list{str} | None, optional):
                Reference covariate name, will be interpreted differently in the
                sub-classes.
            use_re (bool, optional):
                If use the random effects.
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
            prior_gamma_gaussian (numpy.ndarray, optional):
                Direct Gaussian prior for gamma.
            prior_gamma_uniform (numpy.ndarray, optional):
                Direct uniform prior for gamma.
        """
        self.alt_cov = utils.input_cols(alt_cov)
        self.ref_cov = utils.input_cols(ref_cov)
        self.use_re = use_re
        self.use_spline = use_spline

        self.spline_knots_type = spline_knots_type
        self.spline_knots = spline_knots
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
        self.prior_gamma_gaussian = prior_gamma_gaussian
        self.prior_gamma_uniform = prior_gamma_uniform

        self.check_attr()
        self.process_attr()

    def check_attr(self):
        """Check the attributes.
        """
        assert utils.is_cols(self.alt_cov)
        assert utils.is_cols(self.ref_cov)
        if isinstance(self.alt_cov, list):
            assert len(self.alt_cov) <= 2
        if isinstance(self.ref_cov, list):
            assert len(self.ref_cov) <= 2
        assert isinstance(self.use_re, bool)
        assert isinstance(self.use_spline, bool)

        # spline specific
        assert self.spline_knots_type in ['frequency', 'domain']
        assert isinstance(self.spline_knots, np.ndarray)
        assert np.min(self.spline_knots) >= 0.0
        assert np.max(self.spline_knots) <= 1.0
        assert isinstance(self.spline_degree, int)
        assert self.spline_degree >= 0
        assert isinstance(self.spline_l_linear, bool)
        assert isinstance(self.spline_r_linear, bool)

        # priors
        assert (self.prior_spline_monotonicity in ['increasing',
                                                   'decreasing'] or
                self.prior_spline_monotonicity is None)
        assert (self.prior_spline_convexity in ['convex',
                                               'concave'] or
                self.prior_spline_convexity is None)
        assert isinstance(self.prior_spline_num_constraint_points, int)
        assert self.prior_spline_num_constraint_points > 0
        assert utils.is_gaussian_prior(self.prior_spline_maxder_gaussian)
        assert utils.is_gaussian_prior(self.prior_beta_gaussian)
        assert utils.is_gaussian_prior(self.prior_gamma_gaussian)
        assert utils.is_uniform_prior(self.prior_spline_maxder_uniform)
        assert utils.is_uniform_prior(self.prior_beta_uniform)
        assert utils.is_uniform_prior(self.prior_gamma_uniform)

    def process_attr(self):
        """Process attributes.
        """
        # covariates names
        if not isinstance(self.alt_cov, list):
            self.alt_cov = [self.alt_cov]
        if not isinstance(self.ref_cov, list):
            self.ref_cov = [self.ref_cov]

        # spline knots
        self.spline_knots = np.unique(self.spline_knots)
        if np.min(self.spline_knots) > 0.0:
            self.spline_knots = np.insert(self.spline_knots, 0, 0.0)
        if np.max(self.spline_knots) < 1.0:
            self.spline_knots = np.append(self.spline_knots, 1.0)

        # prior information
        self.prior_spline_maxder_gaussian = utils.input_gaussian_prior(
            self.prior_spline_maxder_gaussian, self.spline_knots.size - 1
        )
        self.prior_spline_maxder_uniform = utils.input_uniform_prior(
            self.prior_spline_maxder_uniform, self.spline_knots.size - 1
        )
        self.prior_beta_gaussian = utils.input_gaussian_prior(
            self.prior_beta_gaussian, self.num_x_vars
        )
        self.prior_beta_uniform = utils.input_uniform_prior(
            self.prior_beta_uniform, self.num_x_vars
        )
        self.prior_gamma_gaussian = utils.input_gaussian_prior(
            self.prior_gamma_gaussian, self.num_z_vars
        )
        self.prior_gamma_uniform = utils.input_uniform_prior(
            self.prior_gamma_uniform, self.num_z_vars
        )

    def create_spline(self, data):
        """Create spline given current spline parameters.
        Args:
            data (mrtool.MRData):
                The data frame used for storing the data
        Returns:
            xspline.XSpline
                The spline object.
        """
        # extract covariate
        assert all([cov in data.covs.columns for cov in self.alt_cov])
        assert all([cov in data.covs.columns for cov in self.ref_cov])
        cov = data.covs[self.alt_cov + self.ref_cov].values

        if self.spline_knots_type == 'frequency':
            spline_knots = np.quantile(cov, self.spline_knots)
        else:
            spline_knots = cov.min() + self.spline_knots*(cov.max() - cov.min())

        return xspline.XSpline(spline_knots,
                               self.spline_degree,
                               l_linear=self.spline_l_linear,
                               r_linear=self.spline_r_linear)

    def create_design_mat(self, data):
        """Create design matrix.
        Args:
            data (mrtool.MRData):
                The data frame used for storing the data
        Returns:
            tuple{numpy.ndarray, numpy.ndarray}:
                Return the design matrix for linear cov or spline.
        """
        assert all([cov in data.covs.columns for cov in self.alt_cov])
        assert all([cov in data.covs.columns for cov in self.ref_cov])

        alt_cov = data.covs[self.alt_cov].values
        ref_cov = data.covs[self.ref_cov].values

        spline = self.create_spline(data) if self.use_spline else None

        alt_mat = utils.avg_integral(alt_cov, spline=spline)[:, 1:]
        ref_mat = utils.avg_integral(ref_cov, spline=spline)[:, 1:]

        return alt_mat, ref_mat

    def create_constraint_mat(self, data):
        """Create constraint matrix.
        Args:
            data (mrtool.MRData):
                The data frame used for storing the data

        Returns:
            tuple{numpy.ndarray, numpy.ndarray}:
                Return linear constraints matrix and its uniform prior.
        """
        # initialize the matrix and the value
        c_mat = np.array([]).reshape(0, self.num_x_vars)
        c_val = np.array([]).reshape(2, 0)
        if not self.use_spline:
            return c_mat, c_val

        spline = self.create_spline(data)
        points = np.linspace(spline.knots[0], spline.knots[-1],
                             self.prior_spline_num_constraint_points)
        tmp_val = np.array([[-np.inf], [0.0]])

        # spline monotonicity constraints
        if self.prior_spline_monotonicity is not None:
            sign = 1.0 if self.prior_spline_monotonicity is 'decreasing' \
                else -1.0
            c_mat = np.vstack((c_mat,
                               sign*spline.design_dmat(points, 1)[:, 1:]))
            c_val = np.hstack((c_val,
                               np.repeat(tmp_val, points.size, axis=1)))

        # spline convexity constraints
        if self.prior_spline_convexity is not None:
            sign = 1.0 if self.prior_spline_convexity is 'concave' else -1.0
            c_mat = np.vstack((c_mat,
                               sign*spline.design_dmat(points, 2)[:, 1:]))
            c_val = np.hstack((c_val,
                               np.repeat(tmp_val, points.size, axis=1)))

        # spline maximum derivative constraints
        if not np.isinf(self.prior_spline_maxder_uniform).all():
            c_mat = np.vstack((c_mat, spline.last_dmat()[:, 1:]))
            c_val = np.hstack((c_val, self.prior_spline_maxder_uniform))

        return c_mat, c_val

    def create_regularization_mat(self, data):
        """Create constraint matrix.
        Args:
            data (mrtool.MRData):
                The data frame used for storing the data

        Returns:
            tuple{numpy.ndarray, numpy.ndarray}:
                Return linear regularization matrix and its Gaussian prior.
        """
        r_mat = np.array([]).reshape(0, self.num_x_vars)
        r_val = np.array([]).reshape(2, 0)
        if not self.use_spline:
            return r_mat, r_val

        spline = self.create_spline(data)

        # spline maximum derivative constraints
        if not np.isinf(self.prior_spline_maxder_uniform).all():
            r_mat = np.vstack((r_mat, spline.last_dmat()[:, 1:]))
            r_val = np.hstack((r_val, self.prior_spline_maxder_gaussian))

        return r_mat, r_val

    @property
    def num_x_vars(self):
        if self.use_spline:
            n = self.spline_knots.size - \
                self.spline_l_linear - self.spline_r_linear + \
                self.spline_degree - 2
        else:
            n = 1
        return n

    @property
    def num_z_vars(self):
        if self.use_re:
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
    def __init__(self, *args, use_re_mid_point=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_re_mid_point = use_re_mid_point

    def create_x_mat(self, data):
        """Create design matrix for the fixed effects.

        Args:
            data (mrtool.MRData):
                The data frame used for storing the data

        Returns:
            numpy.ndarray:
                Design matrix for fixed effects.
        """
        alt_mat, ref_mat = self.create_design_mat(data)
        if ref_mat.size == 0:
            return alt_mat
        else:
            return alt_mat - ref_mat

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
            alt_mat = utils.avg_integral(data.covs[self.alt_cov].values)
            ref_mat = utils.avg_integral(data.covs[self.ref_cov].values)
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

        if ref_mat.size == 0:
            def fun(beta):
                return np.log(1.0 + alt_mat.dot(beta))

            def jac_fun(beta):
                return alt_mat/(1.0 + alt_mat.dot(beta)[:, None])
        else:
            def fun(beta):
                return np.log(1.0 + alt_mat.dot(beta)) - \
                    np.log(1.0 + ref_mat.dot(beta))

            def jac_fun(beta):
                return alt_mat/(1.0 + alt_mat.dot(beta)[:, None]) - \
                    ref_mat/(1.0 + ref_mat.dot(beta)[:, None])

        return fun, jac_fun

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

        alt_mat = utils.avg_integral(data.covs[self.alt_cov].values)
        ref_mat = utils.avg_integral(data.covs[self.ref_cov].values)

        if ref_mat.size == 0:
            return alt_mat
        else:
            return alt_mat - ref_mat

    def create_constraint_mat(self, data):
        """Create constraint matrix.
        Overwrite the super class, adding non-negative constraints.
        """
        c_mat, c_val = super().create_constraint_mat(data)
        tmp_val = np.array([[-1.0], [np.inf]])

        if self.use_spline:
            spline = self.create_spline(data)
            points = np.linspace(spline.knots[0], spline.knots[-1],
                                 self.prior_spline_num_constraint_points)
            c_mat = np.vstack((c_mat,
                               spline.design_mat(points)[:, 1:]))
            c_val = np.hstack((c_val,
                               np.repeat(tmp_val, points.size, axis=1)))
        else:
            alt_mat = utils.avg_integral(data.covs[self.alt_cov].values)
            ref_mat = utils.avg_integral(data.covs[self.ref_cov].values)
            cov_mat = np.hstack((alt_mat, ref_mat))
            c_mat = np.vstack((c_mat, np.array([[np.min(cov_mat)],
                                                [np.max(cov_mat)]])))
            c_val = np.hstack((c_val,
                               np.repeat(tmp_val, 2, axis=1)))
        return c_mat, c_val

    @property
    def num_constraints(self):
        num_c = super().num_constraints
        if self.use_spline:
            num_c += self.prior_spline_num_constraint_points
        else:
            num_c += 2
        return num_c
