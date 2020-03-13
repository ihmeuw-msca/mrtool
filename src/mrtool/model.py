# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    Model module for mrtool package.
"""
import numpy as np
from .data import *
from .cov_model import *
from limetr import LimeTr


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

    def check_attr(self):
        """Check the input type of the attributes.
        """
        assert isinstance(self.data, MRData)
        assert isinstance(self.cov_models, list)
        assert all([isinstance(cov_model, CovModel)
                    for cov_model in self.cov_models])
        assert isinstance(self.inlier_pct, float)
        assert (self.inlier_pct >= 0.0) and (self.inlier_pct <= 1.0)

    def create_x_fun(self):
        """Create the fixed effects function, link with limetr.
        """
        # create design matrix for the linear covariates part
        linear_mat = [
            cov_model.create_x_mat(self.data)
            for cov_model in self.linear_cov_models
        ]
        if linear_mat:
            linear_mat = np.hstack(linear_mat)
        else:
            linear_mat = np.array(linear_mat).reshape(self.data.num_obs, 0)
        log_fun = [
            cov_model.create_x_fun(self.data)
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

    def create_z_mat(self):
        """Create the random effects matrix, link with limetr.
        """
        mat = np.hstack([
            cov_model.create_z_mat(self.data)
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

    def fit_model(self):
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

        # create limetr object
        self.lt = LimeTr(n, k_beta, k_gamma,
                         y, x_fun, x_fun_jac, z_mat, S=s,
                         C=c_fun, JC=c_fun_jac, c=c_vec,
                         H=h_fun, JH=h_fun_jac, h=h_vec,
                         uprior=uprior, gprior=gprior,
                         inlier_percentage=self.inlier_pct)

        self.lt.fitModel()
        self.beta_soln = self.lt.beta.copy()
        self.gamma_soln = self.lt.gamma.copy()
        self.w_soln = self.lt.w.copy()
