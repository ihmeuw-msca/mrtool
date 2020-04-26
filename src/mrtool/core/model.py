# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    Model module for mrtool package.
"""
import pandas as pd
from .data import *
from .cov_model import *
from limetr import LimeTr
from copy import deepcopy


class MRBRT:
    """MR-BRT Object
    """
    def __init__(self, data,
                 cov_models=None,
                 inlier_pct=1.0):
        """Constructor of MRBRT.

        Args:
            data (mrtool.MRData):
                Data for meta-regression.
            cov_models (list{mrtool.CovModel} | None, optional):
                A list of covariates models.
            inlier_pct (float, optional):
                A float number between 0 and 1 indicate the percentage of
                inliers.
        """
        self.data = data
        self.cov_models = cov_models if cov_models is not None else [
            LinearCovModel('intercept')]
        self.inlier_pct = inlier_pct

        self.cov_models_dict = {
            cov_model.name: cov_model
            for cov_model in self.cov_models
        }

        # group the linear and log covariates model
        self.linear_cov_models = [
            cov_model
            for cov_model in self.cov_models
            if isinstance(cov_model, LinearCovModel)
        ]
        self.log_cov_models = [
            cov_model
            for cov_model in self.cov_models
            if isinstance(cov_model, LogCovModel)
        ]
        self.cov_models = self.linear_cov_models + self.log_cov_models
        self.linear_cov_model_names = [
            cov_model.name for cov_model in self.linear_cov_models]
        self.log_cov_model_names = [
            cov_model.name for cov_model in self.log_cov_models]
        self.cov_model_names = self.linear_cov_model_names + \
            self.log_cov_model_names

        # fixed effects size and index
        self.x_vars_sizes = {
            cov_model.name: cov_model.num_x_vars
            for cov_model in self.cov_models
        }
        self.num_x_vars = int(sum(self.x_vars_sizes.values()))
        self.num_linear_x_vars = int(sum([
            self.x_vars_sizes[name] for name in self.linear_cov_model_names
        ]))
        self.num_log_x_vars = int(sum([
            self.x_vars_sizes[name] for name in self.log_cov_model_names
        ]))
        x_vars_idx = utils.sizes_to_indices([
            self.x_vars_sizes[name]
            for name in self.cov_model_names
        ])
        self.x_vars_idx = {
            name: x_vars_idx[i]
            for i, name in enumerate(self.cov_model_names)
        }
        self.linear_x_vars_idx = np.arange(self.num_linear_x_vars)
        self.log_x_vars_idx = np.arange(self.num_linear_x_vars,
                                        self.num_x_vars)

        # random effects size and index
        self.z_vars_sizes = {
            cov_model.name: cov_model.num_z_vars
            for cov_model in self.cov_models
        }
        self.num_z_vars = int(sum(self.z_vars_sizes.values()))
        z_vars_idx = utils.sizes_to_indices([
            self.z_vars_sizes[name]
            for name in self.cov_model_names
        ])
        self.z_vars_idx = {
            name: z_vars_idx[i] + self.num_x_vars
            for i, name in enumerate(self.cov_model_names)
        }

        # number of constraints
        self.num_constraints = sum([
            cov_model.num_constraints
            for cov_model in self.cov_models
        ])

        # number of regularizations
        self.num_regularizations = sum([
            cov_model.num_regularizations
            for cov_model in self.cov_models
        ])

        # place holder for the limetr objective
        self.lt = None
        self.beta_soln = None
        self.gamma_soln = None
        self.w_soln = None

    def check_attr(self):
        """Check the input type of the attributes.
        """
        assert isinstance(self.data, MRData)
        assert isinstance(self.cov_models, list)
        assert all([isinstance(cov_model, CovModel)
                    for cov_model in self.cov_models])
        assert isinstance(self.inlier_pct, float)
        assert (self.inlier_pct >= 0.0) and (self.inlier_pct <= 1.0)

    def create_x_fun(self, data=None):
        """Create the fixed effects function, link with limetr.
        """
        data = self.data if data is None else data
        # create design matrix for the linear covariates part
        linear_mat = [
            cov_model.create_x_mat(data)
            for cov_model in self.linear_cov_models
        ]
        if linear_mat:
            linear_mat = np.hstack(linear_mat)
        else:
            linear_mat = np.array(linear_mat).reshape(data.num_obs, 0)
        log_fun = [
            cov_model.create_x_fun(data)
            for cov_model in self.log_cov_models
        ]

        def fun(beta):
            linear_beta = beta[self.linear_x_vars_idx]
            y = linear_mat.dot(linear_beta)
            for i, name in enumerate(self.log_cov_model_names):
                log_beta = beta[self.x_vars_idx[name]]
                y += log_fun[i][0](log_beta)
            return y

        def jac_fun(beta):
            mat = linear_mat
            for i, name in enumerate(self.log_cov_model_names):
                log_beta = beta[self.x_vars_idx[name]]
                mat = np.hstack((mat, log_fun[i][1](log_beta)))
            return mat

        return fun, jac_fun

    def create_z_mat(self, data=None):
        """Create the random effects matrix, link with limetr.
        """
        data = self.data if data is None else data
        mat = np.hstack([
            cov_model.create_z_mat(data)
            for cov_model in self.cov_models
        ])

        return mat

    def create_c_mat(self):
        """Create the constraints matrices.
        """
        num_vars = self.num_x_vars + self.num_z_vars
        c_mat = np.zeros((0, num_vars))
        c_vec = np.zeros((2, 0))

        for cov_model in self.cov_models:
            if cov_model.num_constraints != 0:
                c_mat_sub = np.zeros((cov_model.num_constraints,
                                      num_vars))
                c_mat_sub[:, self.x_vars_idx[cov_model.name]], c_vec_sub = \
                    cov_model.create_constraint_mat(self.data)
                c_mat = np.vstack((c_mat, c_mat_sub))
                c_vec = np.hstack((c_vec, c_vec_sub))

        return c_mat, c_vec

    def create_h_mat(self):
        """Create the regularizer matrices.
        """
        num_vars = self.num_x_vars + self.num_z_vars
        h_mat = np.zeros((0, num_vars))
        h_vec = np.zeros((2, 0))

        for cov_model in self.cov_models:
            if cov_model.num_regularizations != 0:
                h_mat_sub = np.zeros((cov_model.num_regularizations,
                                      num_vars))
                h_mat_sub[:, self.x_vars_idx[cov_model.name]], h_vec_sub = \
                    cov_model.create_regularization_mat(self.data)
                h_mat = np.vstack((h_mat, h_mat_sub))
                h_vec = np.hstack((h_vec, h_vec_sub))

        return h_mat, h_vec

    def create_uprior(self):
        """Create direct uniform prior.
        """
        num_vars = self.num_x_vars + self.num_z_vars
        uprior = np.array([[-np.inf]*num_vars,
                           [np.inf]*num_vars])

        for cov_model in self.cov_models:
            uprior[:, self.x_vars_idx[cov_model.name]] = \
                cov_model.prior_beta_uniform
            uprior[:, self.z_vars_idx[cov_model.name]] = \
                cov_model.prior_gamma_uniform

        return uprior

    def create_gprior(self):
        """Create direct gaussian prior.
        """
        num_vars = self.num_x_vars + self.num_z_vars
        gprior = np.array([[0]*num_vars,
                           [np.inf]*num_vars])

        for cov_model in self.cov_models:
            gprior[:, self.x_vars_idx[cov_model.name]] = \
                cov_model.prior_beta_gaussian
            gprior[:, self.z_vars_idx[cov_model.name]] = \
                cov_model.prior_gamma_gaussian

        return gprior

    def create_lprior(self):
        """Create direct laplace prior.
        """
        num_vars = self.num_x_vars + self.num_z_vars
        lprior = np.array([[0]*num_vars,
                           [np.inf]*num_vars])

        for cov_model in self.cov_models:
            lprior[:, self.x_vars_idx[cov_model.name]] = \
                cov_model.prior_beta_laplace
            lprior[:, self.z_vars_idx[cov_model.name]] = \
                cov_model.prior_gamma_laplace

        return lprior

    def fit_model(self,
                  x0=None,
                  inner_print_level=0,
                  inner_max_iter=20,
                  inner_tol=1e-8,
                  outer_verbose=False,
                  outer_max_iter=100,
                  outer_step_size=1.0,
                  outer_tol=1e-6,
                  normalize_trimming_grad=False):
        """Fitting the model through limetr.
        """
        # dimensions
        n = self.data.study_size.values
        k_beta = self.num_x_vars
        k_gamma = self.num_z_vars

        # data
        y = self.data.obs.values
        s = self.data.obs_se.values

        # create x fun and z mat
        x_fun, x_fun_jac = self.create_x_fun()
        z_mat = self.create_z_mat()

        # priors
        c_mat, c_vec = self.create_c_mat()
        h_mat, h_vec = self.create_h_mat()

        def c_fun(var):
            return c_mat.dot(var)

        def c_fun_jac(var):
            return c_mat

        def h_fun(var):
            return h_mat.dot(var)

        def h_fun_jac(var):
            return h_mat

        uprior = self.create_uprior()
        gprior = self.create_gprior()
        lprior = self.create_lprior()

        # create limetr object
        self.lt = LimeTr(n, k_beta, k_gamma,
                         y, x_fun, x_fun_jac, z_mat, S=s,
                         C=c_fun, JC=c_fun_jac, c=c_vec,
                         H=h_fun, JH=h_fun_jac, h=h_vec,
                         uprior=uprior, gprior=gprior, lprior=lprior,
                         inlier_percentage=self.inlier_pct)

        self.lt.fitModel(x0=x0,
                         inner_print_level=inner_print_level,
                         inner_max_iter=inner_max_iter,
                         inner_tol=inner_tol,
                         outer_verbose=outer_verbose,
                         outer_max_iter=outer_max_iter,
                         outer_step_size=outer_step_size,
                         outer_tol=outer_tol,
                         normalize_trimming_grad=normalize_trimming_grad)
        self.beta_soln = self.lt.beta.copy()
        self.gamma_soln = self.lt.gamma.copy()
        self.w_soln = self.lt.w.copy()

    def sample_soln(self, sample_size=1):
        """Sample solutions.
        """
        if self.lt is None:
            print('Fit the model first!')
            return None, None

        beta_soln_samples, gamma_soln_samples = \
            self.lt.sampleSoln(self.lt, sample_size=sample_size)

        return beta_soln_samples, gamma_soln_samples

    def create_draws(self, data,
                     sample_size=1,
                     beta_samples=None,
                     gamma_samples=None,
                     use_re=False):

        if beta_samples is None or gamma_samples is None:
            beta_samples, gamma_samples = \
                self.sample_soln(sample_size=sample_size)
        else:
            assert beta_samples.shape == (sample_size, self.num_x_vars)
            assert gamma_samples.shape == (sample_size, self.num_z_vars)

        x_fun, x_jac_fun = self.create_x_fun(data=data)
        z_mat = self.create_z_mat(data=data)

        y_samples = np.vstack([
            x_fun(beta_sample)
            for beta_sample in beta_samples
        ])

        if use_re:
            u_samples = np.random.randn(sample_size,
                                        self.num_z_vars)*gamma_samples
            y_samples += u_samples.dot(z_mat.T)

        return y_samples.T


class MRBeRT:
    """Ensemble model of MRBRT.
    """
    def __init__(self, data,
                 ensemble_cov_model,
                 ensemble_knots,
                 cov_models=None,
                 inlier_pct=1.0):
        """Constructor of `MRBeRT`

        Args:
            ensemble_cov_model (mrtool.CovModel):
                Covariates model which will be used with ensemble.
        """
        self.data = data
        self.cov_models = cov_models if cov_models is not None else [
            LinearCovModel('intercept')]
        self.inlier_pct = inlier_pct

        assert isinstance(ensemble_cov_model, CovModel)
        assert ensemble_cov_model.use_spline

        cov_model_tmp = ensemble_cov_model
        self.ensemble_cov_model = cov_model_tmp.name
        self.ensemble_knots = ensemble_knots
        self.num_sub_models = len(ensemble_knots)

        self.sub_models = []
        for knots in self.ensemble_knots:
            ensemble_cov_model = deepcopy(cov_model_tmp)
            ensemble_cov_model.spline_knots = knots.copy()
            ensemble_cov_model.process_attr()
            ensemble_cov_model.check_attr()
            self.sub_models.append(MRBRT(data,
                                         cov_models=[*self.cov_models,
                                                     ensemble_cov_model],
                                         inlier_pct=self.inlier_pct))

        self.weights = np.ones(self.num_sub_models)/self.num_sub_models

    def fit_model(self,
                  x0=None,
                  inner_print_level=0,
                  inner_max_iter=20,
                  inner_tol=1e-8,
                  outer_verbose=False,
                  outer_max_iter=100,
                  outer_step_size=1.0,
                  outer_tol=1e-6,
                  normalize_trimming_grad=False,
                  scores_weights=np.array([1.0, 1.0]),
                  slopes=np.array([2.0, 10.0]),
                  quantiles=np.array([0.4, 0.4])):
        """Fitting the model through limetr.
        """
        for sub_model in self.sub_models:
            sub_model.fit_model(
                x0=x0,
                inner_print_level=inner_print_level,
                inner_max_iter=inner_max_iter,
                inner_tol=inner_tol,
                outer_verbose=outer_verbose,
                outer_max_iter=outer_max_iter,
                outer_step_size=outer_step_size,
                outer_tol=outer_tol,
                normalize_trimming_grad=normalize_trimming_grad
            )

        self.beta_solns = np.vstack([model.beta_soln
                                     for model in self.sub_models])
        self.gamma_solns = np.vstack([model.gamma_soln
                                      for model in self.sub_models])
        self.w_solns = np.vstack([model.w_soln
                                  for model in self.sub_models])

        self.score_model(scores_weights=scores_weights,
                         slopes=slopes,
                         quantiles=quantiles)

    def score_model(self,
                    scores_weights=np.array([1.0, 1.0]),
                    slopes=np.array([2.0, 10.0]),
                    quantiles=np.array([0.4, 0.4])):
        """Score the model by there fitting and variation.
        """
        scores = np.zeros((2, self.num_sub_models))
        for i, sub_model in enumerate(self.sub_models):
            scores[0][i] = utils.score_sub_models_datafit(sub_model)
            scores[1][i] = utils.score_sub_models_variation(
                sub_model, self.ensemble_cov_model, n=3)

        weights = np.zeros(scores.shape)
        for i in range(2):
            weights[i] = utils.nonlinear_trans(
                scores[i],
                slope=slopes[i],
                quantile=quantiles[i]
            )**scores_weights[i]

        weights = np.prod(weights, axis=0)
        self.weights = weights/np.sum(weights)

    def sample_soln(self, sample_size=1):
        """Sample solution.
        """
        try:
            sample_sizes = np.random.multinomial(sample_size, self.weights)
        except:
            print("Fit and evaluate the models first.")
            return None

        beta_samples = []
        gamma_samples = []
        for i, sub_model in enumerate(self.sub_models):
            if sample_sizes[i] != 0:
                sub_beta_samples, sub_gamma_samples = \
                    sub_model.sample_soln(sample_size=sample_sizes[i])
                beta_samples.append(sub_beta_samples)
                gamma_samples.append(sub_gamma_samples)

        beta_samples = np.vstack(beta_samples)
        gamma_samples = np.vstack(gamma_samples)

        sub_model_id = np.repeat(np.arange(self.num_sub_models),
                                 sample_sizes)
        beta_samples = pd.DataFrame({
            'beta_%i'%i: beta_samples[:, i]
            for i in range(beta_samples.shape[1])
        })
        gamma_samples = pd.DataFrame({
            'gamma_%i'%i: gamma_samples[:, i]
            for i in range(gamma_samples.shape[1])
        })
        beta_samples['sub_model_id'] = sub_model_id
        gamma_samples['sub_model_id'] = sub_model_id

        return beta_samples, gamma_samples

    def create_draws(self, data,
                     sample_size=1,
                     beta_samples=None,
                     gamma_samples=None,
                     use_re=False):
        """Create draws.
        """
        if beta_samples is None or gamma_samples is None:
            beta_samples, gamma_samples = \
                self.sample_soln(sample_size=sample_size)
        else:
            assert beta_samples.shape == (sample_size,
                                          self.sub_models[0].num_x_vars + 1)
            assert gamma_samples.shape == (sample_size,
                                           self.sub_models[0].num_z_vars + 1)

        y_samples = []
        for i in range(self.num_sub_models):
            sub_beta_samples = beta_samples[[
                name for name in beta_samples.columns if 'beta' in name
            ]][beta_samples['sub_model_id'] == i].values
            sub_gamma_samples = gamma_samples[[
                name for name in gamma_samples.columns if 'gamma' in name
            ]][gamma_samples['sub_model_id'] == i].values
            if sub_beta_samples.size != 0:
                y_samples.append(self.sub_models[i].create_draws(
                    data,
                    sample_size=sub_beta_samples.shape[0],
                    beta_samples=sub_beta_samples,
                    gamma_samples=sub_gamma_samples,
                    use_re=use_re
                ))
        y_samples = np.hstack(y_samples)

        return y_samples
