# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    Model module for mrtool package.
"""
import numpy as np
from .data import *
from .cov_model import *


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
