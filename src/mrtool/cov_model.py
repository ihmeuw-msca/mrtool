# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~

    Covariates model for `mrtool`.
"""
import numpy as np
from . import utils


class CovModel:
    """Covariates model.
    """
    def __init__(self,
                 alt_cov,
                 ref_cov=None,
                 use_re=False,
                 use_spline=False,
                 spline_knot_type='frequency',
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

        Keyword Args:
            ref_cov (str | list{str} | None, optional):
                Reference covariate name, will be interpreted differently in the
                sub-classes.
            use_re (bool, optional):
                If use the random effects.
            use_spline(bool, optional):
                If use splines.
            spline_knot_type (str, optional):
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
        self.alt_cov = alt_cov
        self.ref_cov = ref_cov
        self.use_re = use_re
        self.use_spline = use_spline

        self.spline_knot_type = spline_knot_type
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
        assert self.spline_knot_type in ['frequency', 'domain']
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
        # spline knots
        self.spline_knots = np.unique(self.spline_knots)
        if np.min(self.spline_knots) > 0.0:
            self.spline_knots = np.insert(self.spline_knots, 0, 0.0)
        if np.max(self.spline_knots) < 1.0:
            self.spline_knots = np.append(self.spline_knots, 1.0)

        # prior information
        self.prior_spline_maxder_gaussian = utils.input_gaussian_prior(
            self.prior_spline_maxder_gaussian, self.num_x_vars
        )
        self.prior_spline_maxder_uniform = utils.input_uniform_prior(
            self.prior_spline_maxder_uniform, self.num_x_vars
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
