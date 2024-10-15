# -*- coding: utf-8 -*-
"""
model
~~~~~

Model module for mrtool package.
"""

from copy import deepcopy
from typing import Sequence

import numpy as np
import pandas as pd
from limetr import LimeTr
from numpy.typing import NDArray

from . import utils
from .cov_model import CovModel
from .data import MRData


class MRBRT:
    """MR-BRT Object"""

    def __init__(
        self,
        data: MRData,
        cov_models: Sequence[CovModel],
        inlier_pct: float = 1.0,
    ):
        """Constructor of MRBRT.

        Parameters
        ----------
        data
            Data for meta-regression.
        cov_models
            A list of covariates models.
        inlier_pct
            A float number between 0 and 1 indicate the percentage of inliers.

        """
        self.data = data
        self.cov_models = cov_models
        self.inlier_pct = inlier_pct
        self.check_input()
        self.cov_model_names = [cov_model.name for cov_model in self.cov_models]
        self.num_cov_models = len(self.cov_models)
        self.cov_names = []
        for cov_model in self.cov_models:
            self.cov_names.extend(cov_model.covs)
        self.num_covs = len(self.cov_names)

        # place holder for the limetr objective
        self.lt: LimeTr
        self.beta_soln: NDArray
        self.gamma_soln: NDArray
        self.u_soln: NDArray
        self.w_soln: NDArray
        self.re_soln: NDArray

    def _infer_shape(self) -> None:
        # add random effects
        if not any([cov_model.use_re for cov_model in self.cov_models]):
            self.cov_models[0].use_re = True
            self.cov_models[0].prior_gamma_uniform = np.array([0.0, 0.0])
            self.cov_models[0]._process_priors()

        # fixed effects size and index
        self.x_vars_sizes = [
            cov_model.num_x_vars for cov_model in self.cov_models
        ]
        self.x_vars_indices = utils.sizes_to_indices(self.x_vars_sizes)
        self.num_x_vars = sum(self.x_vars_sizes)

        # random effects size and index
        self.z_vars_sizes = [
            cov_model.num_z_vars for cov_model in self.cov_models
        ]
        self.z_vars_indices = utils.sizes_to_indices(self.z_vars_sizes)
        self.num_z_vars = sum(self.z_vars_sizes)

        self.num_vars = self.num_x_vars + self.num_z_vars

        # number of constraints
        self.num_constraints = sum(
            [cov_model.num_constraints for cov_model in self.cov_models]
        )

        # number of regularizations
        self.num_regularizations = sum(
            [cov_model.num_regularizations for cov_model in self.cov_models]
        )

    def attach_data(self, data=None):
        """Attach data to cov_model."""
        data = self.data if data is None else data
        # attach data to cov_model
        for cov_model in self.cov_models:
            cov_model.attach_data(data)

    def check_input(self):
        """Check the input type of the attributes."""
        assert isinstance(self.data, MRData)
        assert isinstance(self.cov_models, list)
        assert all(
            [isinstance(cov_model, CovModel) for cov_model in self.cov_models]
        )
        assert (self.inlier_pct >= 0.0) and (self.inlier_pct <= 1.0)

    def get_cov_model(self, name: str) -> CovModel:
        """Choose covariate model with name."""
        index = self.get_cov_model_index(name)
        return self.cov_models[index]

    def get_cov_model_index(self, name: str) -> int:
        """From cov_model name get the index."""
        matching_index = [
            index
            for index, cov_model_name in enumerate(self.cov_model_names)
            if cov_model_name == name
        ]
        num_matching_index = len(matching_index)
        assert (
            num_matching_index == 1
        ), f"Number of matching index is {num_matching_index}."
        return matching_index[0]

    def create_x_fun(self, data=None):
        """Create the fixed effects function, link with limetr."""
        data = self.data if data is None else data
        # create design functions
        design_funs = [
            cov_model.create_x_fun(data) for cov_model in self.cov_models
        ]
        funs, jac_funs = list(zip(*design_funs))

        def x_fun(beta, funs=funs):
            return sum(
                fun(beta[self.x_vars_indices[i]]) for i, fun in enumerate(funs)
            )

        def x_jac_fun(beta, jac_funs=jac_funs):
            return np.hstack(
                [
                    jac_fun(beta[self.x_vars_indices[i]])
                    for i, jac_fun in enumerate(jac_funs)
                ]
            )

        return x_fun, x_jac_fun

    def create_z_mat(self, data=None):
        """Create the random effects matrix, link with limetr."""
        data = self.data if data is None else data
        mat = np.hstack(
            [cov_model.create_z_mat(data) for cov_model in self.cov_models]
        )

        return mat

    def create_c_mat(self):
        """Create the constraints matrices."""
        c_mat = np.zeros((0, self.num_vars))
        c_vec = np.zeros((2, 0))

        for i, cov_model in enumerate(self.cov_models):
            if cov_model.num_constraints != 0:
                c_mat_sub = np.zeros((cov_model.num_constraints, self.num_vars))
                c_mat_sub[:, self.x_vars_indices[i]], c_vec_sub = (
                    cov_model.create_constraint_mat()
                )
                c_mat = np.vstack((c_mat, c_mat_sub))
                c_vec = np.hstack((c_vec, c_vec_sub))

        return c_mat, c_vec

    def create_h_mat(self):
        """Create the regularizer matrices."""
        h_mat = np.zeros((0, self.num_vars))
        h_vec = np.zeros((2, 0))

        for i, cov_model in enumerate(self.cov_models):
            if cov_model.num_regularizations != 0:
                h_mat_sub = np.zeros(
                    (cov_model.num_regularizations, self.num_vars)
                )
                h_mat_sub[:, self.x_vars_indices[i]], h_vec_sub = (
                    cov_model.create_regularization_mat()
                )
                h_mat = np.vstack((h_mat, h_mat_sub))
                h_vec = np.hstack((h_vec, h_vec_sub))

        return h_mat, h_vec

    def create_uprior(self):
        """Create direct uniform prior."""
        uprior = np.array([[-np.inf] * self.num_vars, [np.inf] * self.num_vars])

        for i, cov_model in enumerate(self.cov_models):
            uprior[:, self.x_vars_indices[i]] = cov_model.prior_beta_uniform
            uprior[:, self.z_vars_indices[i] + self.num_x_vars] = (
                cov_model.prior_gamma_uniform
            )

        return uprior

    def create_gprior(self):
        """Create direct gaussian prior."""
        gprior = np.array([[0] * self.num_vars, [np.inf] * self.num_vars])

        for i, cov_model in enumerate(self.cov_models):
            gprior[:, self.x_vars_indices[i]] = cov_model.prior_beta_gaussian
            gprior[:, self.z_vars_indices[i] + self.num_x_vars] = (
                cov_model.prior_gamma_gaussian
            )

        return gprior

    def create_lprior(self):
        """Create direct laplace prior."""
        lprior = np.array([[0] * self.num_vars, [np.inf] * self.num_vars])

        for i, cov_model in enumerate(self.cov_models):
            lprior[:, self.x_vars_indices[i]] = cov_model.prior_beta_laplace
            lprior[:, self.z_vars_indices[i] + self.num_x_vars] = (
                cov_model.prior_gamma_laplace
            )

        return lprior

    def fit_model(self, **fit_options):
        """Fitting the model through limetr.

        Parameters
        ----------
        fit_options
            please check fit arguments in limetr.

        """
        if not all([cov_model.has_data() for cov_model in self.cov_models]):
            self.attach_data()
        self._infer_shape()

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
        # scale z_mat
        z_scale = np.max(np.abs(z_mat), axis=0)
        z_scale[z_scale == 0.0] = 1.0
        z_mat = z_mat / z_scale

        # priors
        c_mat, c_vec = self.create_c_mat()
        h_mat, h_vec = self.create_h_mat()
        c_fun, c_fun_jac = utils.mat_to_fun(c_mat)
        h_fun, h_fun_jac = utils.mat_to_fun(h_mat)

        uprior = self.create_uprior()
        uprior[:, self.num_x_vars : self.num_vars] *= z_scale**2
        gprior = self.create_gprior()
        gprior[:, self.num_x_vars : self.num_vars] *= z_scale**2
        lprior = self.create_lprior()
        lprior[:, self.num_x_vars : self.num_vars] *= z_scale**2

        if np.isneginf(uprior[0]).all() and np.isposinf(uprior[1]).all():
            uprior = None
        if np.isposinf(gprior[1]).all():
            gprior = None
        if np.isposinf(lprior[1]).all():
            lprior = None

        # create limetr object
        self.lt = LimeTr(
            n,
            k_beta,
            k_gamma,
            y,
            x_fun,
            x_fun_jac,
            z_mat,
            S=s,
            C=c_fun,
            JC=c_fun_jac,
            c=c_vec,
            H=h_fun,
            JH=h_fun_jac,
            h=h_vec,
            uprior=uprior,
            gprior=gprior,
            lprior=lprior,
            inlier_percentage=self.inlier_pct,
        )

        self.lt.fit(**fit_options)
        self.lt.Z *= z_scale
        if hasattr(self.lt, "gprior"):
            self.lt.gprior[:, self.lt.idx_gamma] /= z_scale**2
        if hasattr(self.lt, "uprior"):
            self.lt.uprior[:, self.lt.idx_gamma] /= z_scale**2
        if hasattr(self.lt, "lprior"):
            self.lt.lprior[:, self.lt.idx_gamma] /= z_scale**2
        self.lt.gamma /= z_scale**2

        self.beta_soln = self.lt.beta.copy()
        self.gamma_soln = self.lt.gamma.copy()
        self.w_soln = self.lt.w.copy()
        self.u_soln = self.lt.estimate_re()
        self.fe_soln = {
            cov_name: self.beta_soln[
                self.x_vars_indices[self.get_cov_model_index(cov_name)]
            ]
            for cov_name in self.cov_model_names
        }
        self.re_soln = {
            study: self.u_soln[i] for i, study in enumerate(self.data.studies)
        }
        self.re_var_soln = {
            cov_name: self.gamma_soln[
                self.z_vars_indices[self.get_cov_model_index(cov_name)]
            ]
            for cov_name in self.cov_model_names
            if self.cov_models[self.get_cov_model_index(cov_name)].use_re
        }

    def extract_re(self, study_id: NDArray) -> NDArray:
        """Extract the random effect for a given dataset."""
        re = np.vstack(
            [
                self.re_soln[study]
                if study in self.re_soln
                else np.zeros(self.num_z_vars)
                for study in study_id
            ]
        )
        return re

    def predict(
        self,
        data: MRData,
        predict_for_study: bool = False,
        sort_by_data_id: bool = False,
    ) -> NDArray:
        """Create new prediction with existing solution.

        Parameters
        ----------
        data
            MRData object contains the predict data.
        predict_for_study
            If `True`, use the random effects information to prediction for specific
            study. If the `study_id` in `data` do not contain in the fitting data, it
            will assume the corresponding random effects equal to 0.
        sort_by_data_id
            If `True`, will sort the final prediction as the order of the original
            data frame that used to create the `data`. Default to False.

        Returns
        -------
        NDArray
            Predicted outcome array.

        """
        assert data.has_covs(
            self.cov_names
        ), "Prediction data do not have covariates used for fitting."
        x_fun, _ = self.create_x_fun(data=data)
        prediction = x_fun(self.beta_soln)
        if predict_for_study:
            z_mat = self.create_z_mat(data=data)
            re = self.extract_re(data.study_id)
            prediction += np.sum(z_mat * re, axis=1)

        if sort_by_data_id:
            prediction = prediction[np.argsort(data.data_id)]

        return prediction

    def sample_soln(self, sample_size: int = 1) -> tuple[NDArray, NDArray]:
        """Sample solutions.

        Parameters
        ----------
        sample_size
            Number of samples.

        Returns
        -------
        tuple[NDArray, NDArray]
            Return beta samples and gamma samples.

        """
        if self.lt is None:
            raise ValueError("Please fit the model first.")

        beta_soln_samples = self.lt.sample_beta(size=sample_size)
        gamma_soln_samples = np.repeat(
            self.gamma_soln[np.newaxis, :], sample_size, axis=0
        )

        return beta_soln_samples, gamma_soln_samples

    def create_draws(
        self,
        data: MRData,
        beta_samples: NDArray,
        gamma_samples: NDArray,
        random_study: bool = True,
        sort_by_data_id: bool = False,
    ) -> NDArray:
        """Create draws for the given data set.

        Parameters
        ----------
        data
            MRData object contains predict data.
        beta_samples
            Samples of beta.
        gamma_samples
            Samples of gamma.
        random_study
            If `True` the draws will include uncertainty from study heterogeneity.
        sort_by_data_id
            If `True`, will sort the final prediction as the order of the original
            data frame that used to create the `data`. Default to False.

        Returns
        -------
        NDArray
            Returns outcome sample matrix.

        """
        sample_size = beta_samples.shape[0]
        assert beta_samples.shape == (sample_size, self.num_x_vars)
        assert gamma_samples.shape == (sample_size, self.num_z_vars)

        x_fun, x_jac_fun = self.create_x_fun(data=data)
        z_mat = self.create_z_mat(data=data)

        y_samples = np.vstack(
            [x_fun(beta_sample) for beta_sample in beta_samples]
        )

        if random_study:
            u_samples = np.random.randn(sample_size, self.num_z_vars) * np.sqrt(
                gamma_samples
            )
            y_samples += u_samples.dot(z_mat.T)
        else:
            re = self.extract_re(data.study_id)
            y_samples += np.sum(z_mat * re, axis=1)

        if sort_by_data_id:
            y_samples = y_samples[:, np.argsort(data.data_id)]

        return y_samples.T

    def summary(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the summary data frame."""
        fe = pd.DataFrame(utils.ravel_dict(self.fe_soln), index=[0])
        re_var = pd.DataFrame(utils.ravel_dict(self.re_var_soln), index=[0])

        return fe, re_var


class MRBeRT:
    """Ensemble model of MRBRT."""

    def __init__(
        self,
        data: MRData,
        ensemble_cov_model: CovModel,
        ensemble_knots: NDArray,
        cov_models: list[CovModel] | None = None,
        inlier_pct: float = 1.0,
    ):
        """Constructor of `MRBeRT`

        Parameters
        ----------
        data
            Data for meta-regression.
        ensemble_cov_model
            Covariates model which will be used with ensemble.
        cov_models
            Other covariate models, assume to be mutual exclusive with ensemble_cov_mdoel.
        inlier_pct
            A float number between 0 and 1 indicate the percentage of inliers.

        """
        self.data = data
        self.cov_models = cov_models if cov_models is not None else []
        self.inlier_pct = inlier_pct

        assert isinstance(ensemble_cov_model, CovModel)
        assert ensemble_cov_model.use_spline

        cov_model_tmp = ensemble_cov_model
        self.ensemble_cov_model_name = cov_model_tmp.name
        self.ensemble_knots = ensemble_knots
        self.num_sub_models = len(ensemble_knots)

        self.sub_models = []
        for knots in self.ensemble_knots:
            ensemble_cov_model = deepcopy(cov_model_tmp)
            ensemble_cov_model.spline_knots_template = knots.copy()
            self.sub_models.append(
                MRBRT(
                    data,
                    cov_models=[*self.cov_models, ensemble_cov_model],
                    inlier_pct=self.inlier_pct,
                )
            )

        self.weights = np.ones(self.num_sub_models) / self.num_sub_models

    def _infer_shape(self) -> None:
        # inherent the dimension variable
        self.num_x_vars = self.sub_models[0].num_x_vars
        self.num_z_vars = self.sub_models[0].num_z_vars
        self.num_vars = self.sub_models[0].num_vars
        self.num_constraints = self.sub_models[0].num_constraints
        self.num_regularizations = self.sub_models[0].num_regularizations
        self.num_cov_models = self.sub_models[0].num_cov_models

    def fit_model(
        self,
        scores_weights=np.array([1.0, 1.0]),
        slopes=np.array([2.0, 10.0]),
        quantiles=np.array([0.4, 0.4]),
        **fit_options,
    ):
        """Fitting the model through limetr."""
        for sub_model in self.sub_models:
            sub_model.fit_model(**fit_options)

        self._infer_shape()

        self.score_model(
            scores_weights=scores_weights, slopes=slopes, quantiles=quantiles
        )

        self.w_soln = np.vstack(
            [sub_model.w_soln for sub_model in self.sub_models]
        ).T.dot(self.weights)

    def score_model(
        self,
        scores_weights=np.array([1.0, 1.0]),
        slopes=np.array([2.0, 10.0]),
        quantiles=np.array([0.4, 0.4]),
    ):
        """Score the model by there fitting and variation."""
        scores = np.zeros((2, self.num_sub_models))
        for i, sub_model in enumerate(self.sub_models):
            scores[0][i] = score_sub_models_datafit(sub_model)
            scores[1][i] = score_sub_models_variation(
                sub_model, self.ensemble_cov_model_name, n=3
            )

        weights = np.zeros(scores.shape)
        for i in range(2):
            weights[i] = (
                utils.nonlinear_trans(
                    scores[i], slope=slopes[i], quantile=quantiles[i]
                )
                ** scores_weights[i]
            )

        weights = np.prod(weights, axis=0)
        self.weights = weights / np.sum(weights)

    def sample_soln(
        self, sample_size: int = 1
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Sample solution."""
        sample_sizes = np.random.multinomial(sample_size, self.weights)

        beta_samples = []
        gamma_samples = []
        for i, sub_model in enumerate(self.sub_models):
            if sample_sizes[i] != 0:
                sub_beta_samples, sub_gamma_samples = sub_model.sample_soln(
                    sample_size=sample_sizes[i]
                )
            else:
                sub_beta_samples = np.array([]).reshape(0, sub_model.num_x_vars)
                sub_gamma_samples = np.array([]).reshape(
                    0, sub_model.num_z_vars
                )
            beta_samples.append(sub_beta_samples)
            gamma_samples.append(sub_gamma_samples)

        return beta_samples, gamma_samples

    def predict(
        self,
        data: MRData,
        predict_for_study: bool = False,
        sort_by_data_id: bool = False,
        return_avg: bool = True,
    ) -> NDArray:
        """Create new prediction with existing solution.

        Parameters
        ----------
        return_avg
            When it is `True`, the function will return an average prediction based on the score,
            and when it is `False` the function will return a list of predictions from all groups.

        """
        prediction = np.vstack(
            [
                sub_model.predict(
                    data,
                    predict_for_study=predict_for_study,
                    sort_by_data_id=sort_by_data_id,
                )
                for sub_model in self.sub_models
            ]
        )

        if return_avg:
            prediction = prediction.T.dot(self.weights)

        return prediction

    def create_draws(
        self,
        data: MRData,
        beta_samples: list[NDArray],
        gamma_samples: list[NDArray],
        random_study: bool = True,
        sort_by_data_id: bool = False,
    ) -> NDArray:
        """Create draws.
        For function description please check `create_draws` for `MRBRT`.
        """
        sample_sizes = [
            sub_beta_samples.shape[0] for sub_beta_samples in beta_samples
        ]
        for i in range(self.num_sub_models):
            assert beta_samples[i].shape == (
                sample_sizes[i],
                self.sub_models[0].num_x_vars,
            )
            assert gamma_samples[i].shape == (
                sample_sizes[i],
                self.sub_models[0].num_z_vars,
            )

        y_samples = []
        for i in range(self.num_sub_models):
            sub_beta_samples = beta_samples[i]
            sub_gamma_samples = gamma_samples[i]
            if sub_beta_samples.size != 0:
                y_samples.append(
                    self.sub_models[i].create_draws(
                        data,
                        beta_samples=sub_beta_samples,
                        gamma_samples=sub_gamma_samples,
                        random_study=random_study,
                        sort_by_data_id=sort_by_data_id,
                    )
                )
        y_samples = np.hstack(y_samples)

        return y_samples

    def summary(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create summary data frame."""
        summary_list = [sub_model.summary() for sub_model in self.sub_models]
        fe = pd.concat([summary_list[i][0] for i in range(self.num_sub_models)])
        fe.loc[self.num_sub_models] = fe.values.T.dot(self.weights)
        fe.reset_index(inplace=True, drop=True)
        fe.insert(
            0,
            "model_id",
            np.hstack((np.arange(self.num_sub_models), "average")),
        )
        fe["weights"] = np.hstack((self.weights, np.nan))

        re_var = pd.concat(
            [summary_list[i][1] for i in range(self.num_sub_models)]
        )
        re_var.loc[self.num_sub_models] = re_var.values.T.dot(self.weights)
        re_var.reset_index(inplace=True, drop=True)
        re_var.insert(
            0,
            "model_id",
            np.hstack((np.arange(self.num_sub_models), "average")),
        )
        re_var["weights"] = np.hstack((self.weights, np.nan))

        return fe, re_var


def score_sub_models_datafit(mr: MRBRT):
    """score the result of mrbert"""
    if mr.lt.soln is None:
        raise ValueError("Must optimize MRBRT first.")

    return -mr.lt.objective(mr.lt.soln)


def score_sub_models_variation(
    mr: MRBRT, ensemble_cov_model_name: str, n: int = 1
) -> float:
    """score the result of mrbert"""
    if mr.lt.soln is None:
        raise ValueError("Must optimize MRBRT first.")

    index = mr.get_cov_model_index(ensemble_cov_model_name)
    cov_model = mr.cov_models[index]
    spline = cov_model.spline
    x = np.linspace(spline.knots[0], spline.knots[-1], 201)
    i = 0 if cov_model.use_spline_intercept else 1
    dmat = spline.design_dmat(x, n)[:, i:]
    d = dmat.dot(mr.beta_soln[mr.x_vars_indices[index]])
    return -np.mean(np.abs(d))
