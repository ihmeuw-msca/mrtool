# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    Model module for mrtool package.
"""
from typing import List, Dict, Union, Tuple
from copy import deepcopy
import numpy as np
from limetr import LimeTr
from .data import MRData
from .cov_model import CovModel
from . import utils


class MRBRT:
    """MR-BRT Object
    """
    def __init__(self, data: MRData,
                 cov_models: List[CovModel],
                 inlier_pct: float = 1.0):
        """Constructor of MRBRT.

        Args:
            data (MRData): Data for meta-regression.
            cov_models (List[CovModel]): A list of covariates models.
            inlier_pct (float, optional):
                A float number between 0 and 1 indicate the percentage of
                inliers.
        """
        self.data = data
        self.cov_models = cov_models
        self.inlier_pct = inlier_pct
        self.check_input()
        self.cov_model_names = [
            cov_model.name for cov_model in self.cov_models
        ]
        self.num_cov_models = len(self.cov_models)

        # attach data to cov_model
        for cov_model in self.cov_models:
            cov_model.attach_data(self.data)

        # fixed effects size and index
        self.x_vars_sizes = [cov_model.num_x_vars for cov_model in self.cov_models]
        self.x_vars_indices = utils.sizes_to_indices(self.x_vars_sizes)
        self.num_x_vars = sum(self.x_vars_sizes)

        # random effects size and index
        self.z_vars_sizes = [cov_model.num_z_vars for cov_model in self.cov_models]
        self.z_vars_indices = utils.sizes_to_indices(self.z_vars_sizes)
        self.num_z_vars = sum(self.z_vars_sizes)

        self.num_vars = self.num_x_vars + self.num_z_vars

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
        self.u_soln = None
        self.w_soln = None
        self.re_soln = None

    def check_input(self):
        """Check the input type of the attributes.
        """
        assert isinstance(self.data, MRData)
        assert isinstance(self.cov_models, list)
        assert all([isinstance(cov_model, CovModel)
                    for cov_model in self.cov_models])
        assert (self.inlier_pct >= 0.0) and (self.inlier_pct <= 1.0)

    def get_cov_model(self, name: str) -> CovModel:
        """Choose covariate model with name.
        """
        matching_index = [index for index, cov_model_name in enumerate(self.cov_models)
                          if cov_model_name == name]
        num_matching_index = len(matching_index)
        assert num_matching_index == 1, f"Number of matching index is {num_matching_index}."
        return self.cov_models[matching_index[0]]

    def create_x_fun(self, data=None):
        """Create the fixed effects function, link with limetr.
        """
        data = self.data if data is None else data
        # create design functions
        design_funs = [
            cov_model.create_x_fun(data)
            for cov_model in self.cov_models
        ]
        funs, jac_funs = list(zip(*design_funs))

        def x_fun(beta, funs=funs):
            return sum(fun(beta[self.x_vars_indices[i]])
                       for i, fun in enumerate(funs))

        def x_jac_fun(beta, jac_funs=jac_funs):
            return np.hstack([jac_fun(beta[self.x_vars_indices[i]])
                              for i, jac_fun in enumerate(jac_funs)])

        return x_fun, x_jac_fun

    def create_z_mat(self, data=None):
        """Create the random effects matrix, link with limetr.
        """
        data = self.data if data is None else data
        mat = np.hstack([cov_model.create_z_mat(data)
                         for cov_model in self.cov_models])

        return mat

    def create_c_mat(self):
        """Create the constraints matrices.
        """
        c_mat = np.zeros((0, self.num_vars))
        c_vec = np.zeros((2, 0))

        for i, cov_model in enumerate(self.cov_models):
            if cov_model.num_constraints != 0:
                c_mat_sub = np.zeros((cov_model.num_constraints, self.num_vars))
                c_mat_sub[:, self.x_vars_indices[i]], c_vec_sub = cov_model.create_constraint_mat()
                c_mat = np.vstack((c_mat, c_mat_sub))
                c_vec = np.hstack((c_vec, c_vec_sub))

        return c_mat, c_vec

    def create_h_mat(self):
        """Create the regularizer matrices.
        """
        h_mat = np.zeros((0, self.num_vars))
        h_vec = np.zeros((2, 0))

        for i, cov_model in enumerate(self.cov_models):
            if cov_model.num_regularizations != 0:
                h_mat_sub = np.zeros((cov_model.num_regularizations, self.num_vars))
                h_mat_sub[:, self.x_vars_indices[i]], h_vec_sub = cov_model.create_regularization_mat()
                h_mat = np.vstack((h_mat, h_mat_sub))
                h_vec = np.hstack((h_vec, h_vec_sub))

        return h_mat, h_vec

    def create_uprior(self):
        """Create direct uniform prior.
        """
        uprior = np.array([[-np.inf]*self.num_vars,
                           [np.inf]*self.num_vars])

        for i, cov_model in enumerate(self.cov_models):
            uprior[:, self.x_vars_indices[i]] = cov_model.prior_beta_uniform
            uprior[:, self.z_vars_indices[i] + self.num_x_vars] = cov_model.prior_gamma_uniform

        return uprior

    def create_gprior(self):
        """Create direct gaussian prior.
        """
        gprior = np.array([[0]*self.num_vars,
                           [np.inf]*self.num_vars])

        for i, cov_model in enumerate(self.cov_models):
            gprior[:, self.x_vars_indices[i]] = cov_model.prior_beta_gaussian
            gprior[:, self.z_vars_indices[i] + self.num_x_vars] = cov_model.prior_gamma_gaussian

        return gprior

    def create_lprior(self):
        """Create direct laplace prior.
        """
        lprior = np.array([[0]*self.num_vars,
                           [np.inf]*self.num_vars])

        for i, cov_model in enumerate(self.cov_models):
            lprior[:, self.x_vars_indices[i]] = cov_model.prior_beta_laplace
            lprior[:, self.z_vars_indices[i] + self.num_x_vars] = cov_model.prior_gamma_laplace

        return lprior

    def fit_model(self, **fit_options: Dict):
        """Fitting the model through limetr.
        """
        # dimensions
        n = self.data.study_sizes
        k_beta = self.num_x_vars
        k_gamma = self.num_z_vars

        # data
        y = self.data.obs
        s = self.data.obs_se

        # create x fun and z mat
        x_fun, x_fun_jac = self.create_x_fun()
        z_mat = self.create_z_mat()

        # priors
        c_mat, c_vec = self.create_c_mat()
        h_mat, h_vec = self.create_h_mat()
        c_fun, c_fun_jac = utils.mat_to_fun(c_mat)
        h_fun, h_fun_jac = utils.mat_to_fun(h_mat)

        uprior = self.create_uprior()
        gprior = self.create_gprior()
        lprior = self.create_lprior()

        if np.isneginf(uprior[0]).all() and np.isposinf(uprior[1]).all():
            uprior = None
        if np.isposinf(gprior[1]).all():
            gprior = None
        if np.isposinf(lprior[1]).all():
            lprior = None

        # create limetr object
        self.lt = LimeTr(n, k_beta, k_gamma,
                         y, x_fun, x_fun_jac, z_mat, S=s,
                         C=c_fun, JC=c_fun_jac, c=c_vec,
                         H=h_fun, JH=h_fun_jac, h=h_vec,
                         uprior=uprior, gprior=gprior, lprior=lprior,
                         inlier_percentage=self.inlier_pct)

        self.lt.fitModel(**fit_options)
        self.beta_soln = self.lt.beta.copy()
        self.gamma_soln = self.lt.gamma.copy()
        self.w_soln = self.lt.w.copy()
        self.u_soln = self.lt.estimateRE()
        self.re_soln = {
            study: self.u_soln[i]
            for i, study in enumerate(self.data.studies)
        }

    def extract_re(self, study_id: np.ndarray) -> np.ndarray:
        """Extract the random effect for a given dataset.
        """
        re = np.vstack([
            self.re_soln[study] if study in self.re_soln else np.zeros(self.num_z_vars)
            for study in study_id
        ])
        return re

    def predict(self, data: MRData,
                predict_for_study=False) -> np.ndarray:
        """Create new prediction with existing solution.
        """
        assert data.has_covs(self.cov_model_names), "Prediction data do not have covariates used for fitting."
        x_fun, _ = self.create_x_fun(data=data)
        prediction = x_fun(self.beta_soln)
        if predict_for_study:
            z_mat = self.create_z_mat(data=data)
            re = self.extract_re(data.study_id)
            prediction += np.sum(z_mat*re, axis=1)

        return prediction

    def sample_soln(self,
                    sample_size: int = 1,
                    sim_prior: bool = True,
                    sim_re: bool = True,
                    print_level: bool = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Sample solutions.
        """
        if self.lt is None:
            raise ValueError('Please fit the model first.')

        beta_soln_samples, gamma_soln_samples = \
            self.lt.sampleSoln(self.lt,
                               sample_size=sample_size,
                               sim_prior=sim_prior,
                               sim_re=sim_re,
                               print_level=print_level)

        return beta_soln_samples, gamma_soln_samples

    def create_draws(self,
                     data: MRData,
                     beta_samples: Union[np.ndarray, None] = None,
                     gamma_samples: Union[np.ndarray, None] = None,
                     sample_soln_options: Union[dict, None] = None,
                     random_study: bool = True) -> np.ndarray:
        """Create draws for the given dataset.
        """
        sample_soln_options = dict() if sample_soln_options is None else sample_soln_options
        if beta_samples is None or gamma_samples is None:
            beta_samples, gamma_samples = self.sample_soln(**sample_soln_options)

        sample_size = beta_samples.shape[0]
        assert beta_samples.shape == (sample_size, self.num_x_vars)
        assert gamma_samples.shape == (sample_size, self.num_z_vars)

        x_fun, x_jac_fun = self.create_x_fun(data=data)
        z_mat = self.create_z_mat(data=data)

        y_samples = np.vstack([x_fun(beta_sample) for beta_sample in beta_samples])

        if random_study:
            u_samples = np.random.randn(sample_size, self.num_z_vars)*np.sqrt(gamma_samples)
            y_samples += u_samples.dot(z_mat.T)
        else:
            re = self.extract_re(data.study_id)
            y_samples += np.sum(z_mat*re, axis=1)

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
        self.cov_models = cov_models if cov_models is not None else []
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
            ensemble_cov_model.spline_knots_template = knots.copy()
            ensemble_cov_model.attach_data(data)
            self.sub_models.append(MRBRT(data,
                                         cov_models=[*self.cov_models, ensemble_cov_model],
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
            sub_model.fit_model(**dict(
                x0=x0,
                inner_print_level=inner_print_level,
                inner_max_iter=inner_max_iter,
                inner_tol=inner_tol,
                outer_verbose=outer_verbose,
                outer_max_iter=outer_max_iter,
                outer_step_size=outer_step_size,
                outer_tol=outer_tol,
                normalize_trimming_grad=normalize_trimming_grad
            ))

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

    def sample_soln(self,
                    sample_size: int = 1,
                    sim_prior: bool = True,
                    sim_re: bool = True,
                    print_level: bool = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Sample solution.
        """
        sample_sizes = np.random.multinomial(sample_size, self.weights)

        beta_samples = []
        gamma_samples = []
        for i, sub_model in enumerate(self.sub_models):
            if sample_sizes[i] != 0:
                sub_beta_samples, sub_gamma_samples = \
                    sub_model.sample_soln(sample_size=sample_sizes[i],
                                          sim_prior=sim_prior,
                                          sim_re=sim_re,
                                          print_level=print_level)
            else:
                sub_beta_samples = np.array([]).reshape(0, sub_model.num_x_vars)
                sub_gamma_samples = np.array([]).reshape(0, sub_model.num_z_vars)
            beta_samples.append(sub_beta_samples)
            gamma_samples.append(sub_gamma_samples)

        return beta_samples, gamma_samples

    def create_draws(self,
                     data: MRData,
                     beta_samples: Union[List[np.ndarray], None] = None,
                     gamma_samples: Union[List[np.ndarray], None] = None,
                     sample_soln_options: Union[dict, None] = None,
                     random_study: bool = True) -> np.ndarray:
        """Create draws.
        """
        if beta_samples is None or gamma_samples is None:
            beta_samples, gamma_samples = self.sample_soln(**sample_soln_options)

        sample_sizes = [sub_beta_samples.shape[0] for sub_beta_samples in beta_samples]
        for i in range(self.num_sub_models):
            assert beta_samples[i].shape == (sample_sizes[i], self.sub_models[0].num_x_vars)
            assert gamma_samples[i].shape == (sample_sizes[i], self.sub_models[0].num_z_vars)

        y_samples = []
        for i in range(self.num_sub_models):
            sub_beta_samples = beta_samples[i]
            sub_gamma_samples = gamma_samples[i]
            if sub_beta_samples.size != 0:
                y_samples.append(self.sub_models[i].create_draws(
                    data,
                    beta_samples=sub_beta_samples,
                    gamma_samples=sub_gamma_samples,
                    random_study=random_study
                ))
        y_samples = np.hstack(y_samples)

        return y_samples
