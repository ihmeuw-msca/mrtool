# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    `model` module of the `crosswalk` package.
"""
from typing import List
import warnings
import numpy as np
import pandas as pd
import limetr
from limetr import LimeTr
from xspline import XSpline
from mrtool.xwalk.data import XData
from mrtool.xwalk.utils import default_input, sizes_to_indices
from mrtool.core.cov_model import LinearCovModel


class CWModel:
    """Cross Walk model.
    """

    def __init__(self, xdata,
                 obs_type='diff_log',
                 cov_models=None,
                 gold_dorm=None,
                 order_prior=None,
                 use_random_intercept=True,
                 prior_gamma_uniform=None,
                 prior_gamma_gaussian=None):
        """Constructor of CWModel.
        Args:
            xdata (XData):
                Data for cross walk.
            obs_type (str, optional):
                Type of observation can only be chosen from `'diff_log'` and
                `'diff_logit'`.
            cov_models (list{crosswalk.CovModel}):
                A list of covariate models for the definitions/methods
            gold_dorm (str | None, optional):
                Gold standard definition/method.
            order_prior (list{list{str}} | None, optional):
                Order priors between different definitions.
            use_random_intercept (bool, optional):
                If ``True``, use random intercept.
            prior_gamma_uniform (Tuple[float, float], optional):
                If not ``None``, use it as the bound of gamma.
            prior_gamma_gaussian (Tuple[float, float], optional):
                If not ``None``, use it as the gaussian prior of gamma.
        """
        self.xdata = xdata
        self.obs_type = obs_type
        self.cov_models = default_input(cov_models,
                                        [LinearCovModel('intercept')])
        self.gold_dorm = default_input(gold_dorm, xdata.max_ref_dorm)
        self.order_prior = order_prior
        self.use_random_intercept = use_random_intercept
        if self.xdata.num_studies == 0 and self.use_random_intercept:
            warnings.warn("Must have study_id to use random intercept."
                          " Reset use_random_intercept to False.")
            self.use_random_intercept = False

        # check input
        self.check()

        # create function for prediction
        if self.obs_type == 'diff_log':
            def obs_fun(x):
                return np.log(x)

            def obs_inv_fun(y):
                return np.exp(y)
        else:
            def obs_fun(x):
                return np.log(x/(1.0 - x))

            def obs_inv_fun(y):
                return 1.0/(1.0 + np.exp(-y))

        self.obs_fun = obs_fun
        self.obs_inv_fun = obs_inv_fun

        # variable names
        self.vars = [dorm for dorm in self.xdata.unique_dorms]

        # dimensions
        self.num_vars_per_dorm = sum([model.num_x_vars for model in self.cov_models])
        self.num_vars = self.num_vars_per_dorm*self.xdata.num_dorms

        # indices for easy access the variables
        var_sizes = np.array([self.num_vars_per_dorm]*self.xdata.num_dorms)
        var_idx = sizes_to_indices(var_sizes)
        self.var_idx = {
            var: var_idx[i]
            for i, var in enumerate(self.vars)
        }

        # create design matrix
        self.relation_mat = self.create_relation_mat()
        self._check_relation_mat()
        self.cov_mat = self.create_cov_mat()
        self._assert_covs_independent()
        self.design_mat = self.create_design_mat()
        self._assert_rank_efficient()
        self.constraint_mat = self.create_constraint_mat()

        # gamma bounds
        self.prior_gamma_uniform = np.array(
            [0.0, np.inf]) if prior_gamma_uniform is None else np.array(prior_gamma_uniform)
        if not self.use_random_intercept:
            self.prior_gamma_uniform = np.zeros(2)
        if self.prior_gamma_uniform[0] < 0.0:
            warnings.warn("Lower bound of gamma has to be non-negative, reset it to zero.")
            self.prior_gamma_uniform[0] = 0.0

        # gamma Gaussian prior
        self.prior_gamma_gaussian = np.array(
            [0.0, np.inf]) if prior_gamma_gaussian is None else np.array(prior_gamma_gaussian)
        if not self.use_random_intercept:
            self.prior_gamma_gaussian = np.array([0.0, np.inf])

        # beta bounds
        uprior = np.repeat(np.array([[-np.inf], [np.inf]]), self.num_vars, axis=1)
        for i, cov_model in enumerate(self.cov_models):
            for dorm in self.xdata.dorms:
                uprior[:, self.var_idx[dorm][i]] = cov_model.prior_beta_uniform
        uprior[:, self.var_idx[self.gold_dorm]] = 0.0
        self.prior_beta_uniform = uprior

        # beta Gaussian prior
        gprior = np.repeat(np.array([[0.0], [np.inf]]), self.num_vars, axis=1)
        for i, cov_model in enumerate(self.cov_models):
            for dorm in self.xdata.dorms:
                gprior[:, self.var_idx[dorm][i]] = cov_model.prior_beta_gaussian
        gprior[:, self.var_idx[self.gold_dorm]] = np.array([[0.0], [np.inf]])
        self.prior_beta_gaussian = gprior

        # current covaraites names
        self.cov_names = []
        for cov_model in self.cov_models:
            self.cov_names.extend(cov_model.covs)
        self.num_covs = len(self.cov_names)

        # place holder for the solutions
        self.beta = None
        self.beta_sd = None
        self.gamma = None
        self.fixed_vars = None
        self.random_vars = None

    def check(self):
        """Check input type, dimension and values.
        """
        assert isinstance(self.xdata, XData)
        assert self.obs_type in ['diff_log', 'diff_logit'], \
            "Unsupport observation type"
        assert isinstance(self.cov_models, list)
        assert all([isinstance(model, LinearCovModel) for model in self.cov_models])

        assert self.gold_dorm in self.xdata.unique_dorms

        assert self.order_prior is None or isinstance(self.order_prior, list)

    def _assert_covs_independent(self):
        """Check if the covariates are independent.
        """
        rank = np.linalg.matrix_rank(self.cov_mat)
        if rank < self.cov_mat.shape[1]:
            raise ValueError("Covariates are collinear, that is, some covariate column is a linear combination of "
                             "some of the other columns. Please check them carefully.")

    def _assert_rank_efficient(self):
        """Check the rank of the design matrix.
        """
        rank = np.linalg.matrix_rank(self.design_mat)
        num_unknowns = self.num_vars_per_dorm*(self.xdata.num_dorms - 1)
        if rank < num_unknowns:
            raise ValueError(f"Not enough information in the data to recover parameters."
                             f"Number of effective data points is {rank} and number of unknowns is {num_unknowns}."
                             f"Please include more effective data or reduce the number of covariates.")

    def create_relation_mat(self, xdata=None):
        """Creating relation matrix.

        Args:
            xdata (data.xdata | None, optional):
                Optional data set, if None, use `self.xdata`.

        Returns:
            numpy.ndarray:
                Returns relation matrix with 1 encode alternative definition
                and -1 encode reference definition.
        """
        xdata = default_input(xdata, default=self.xdata)
        assert isinstance(xdata, XData)

        relation_mat = np.zeros((xdata.num_obs, xdata.num_dorms))
        for i, dorms in enumerate(xdata.alt_dorms):
            for dorm in dorms:
                relation_mat[i, xdata.dorm_idx[dorm]] += 1.0

        for i, dorms in enumerate(xdata.ref_dorms):
            for dorm in dorms:
                relation_mat[i, xdata.dorm_idx[dorm]] -= 1.0

        return relation_mat

    def _check_relation_mat(self):
        """Check relation matrix, detect unused dorms.
        """
        col_scales = np.max(np.abs(self.relation_mat), axis=0)
        unused_dorms = [self.xdata.unique_dorms[i]
                        for i, scale in enumerate(col_scales) if scale == 0.0]
        if len(unused_dorms) != 0:
            raise ValueError(f"{unused_dorms} appears to be unused, most likely it is (they are) "
                             f"in both alt_dorms and ref_dorms at the same time for all its (their) "
                             f"appearance. Please remove {unused_dorms} from alt_dorms and ref_dorms.")

    def create_cov_mat(self, xdata=None):
        """Create covariates matrix for definitions/methods model.

        Args:
            xdata (data.xdata | None, optional):
                Optional data set, if None, use `self.xdata`.

        Returns:
            numpy.ndarray:
                Returns covarites matrix.
        """
        xdata = default_input(xdata, default=self.xdata)
        assert isinstance(xdata, XData)

        return np.hstack([cov_model.create_design_mat(xdata)[0]
                          for cov_model in self.cov_models])

    def create_design_mat(self,
                          xdata=None,
                          relation_mat=None,
                          cov_mat=None):
        """Create linear design matrix.

        Args:
            xdata (data.xdata | None, optional):
                Optional data set, if None, use `self.xdata`.
            relation_mat (numpy.ndarray | None, optional):
                Optional relation matrix, if None, use `self.relation_mat`
            cov_mat (numpy.ndarray | None, optional):
                Optional covariates matrix, if None, use `self.cov_mat`

        Returns:
            numpy.ndarray:
                Returns linear design matrix.
        """
        xdata = default_input(xdata, default=self.xdata)
        relation_mat = default_input(relation_mat, default=self.relation_mat)
        cov_mat = default_input(cov_mat, default=self.cov_mat)

        mat = (relation_mat.ravel()[:, None] *
               np.repeat(cov_mat, xdata.num_dorms, axis=0)).reshape(xdata.num_obs, self.num_vars)

        return mat

    def create_constraint_mat(self):
        """Create constraint matrix.

        Returns:
            numpy.ndarray:
                Return constraints matrix.
        """
        mat = np.array([]).reshape(0, self.num_vars)
        if self.order_prior is not None:
            dorm_constraint_mat = []
            cov_mat = self.cov_mat
            min_cov_mat = np.min(cov_mat, axis=0)
            max_cov_mat = np.max(cov_mat, axis=0)

            if np.allclose(min_cov_mat, max_cov_mat):
                design_mat = min_cov_mat[None, :]
            else:
                design_mat = np.vstack((
                    min_cov_mat,
                    max_cov_mat
                ))
            for p in self.order_prior:
                sub_mat = np.zeros((design_mat.shape[0], self.num_vars))
                sub_mat[:, self.var_idx[p[0]]] = design_mat
                sub_mat[:, self.var_idx[p[1]]] = -1*design_mat
                dorm_constraint_mat.append(sub_mat)
            dorm_constraint_mat = np.vstack(dorm_constraint_mat)
            mat = np.vstack((mat, dorm_constraint_mat))

        if mat.size == 0:
            return None
        return mat

    def fit(self,
            max_iter=100,
            inlier_pct=1.0,
            outer_max_iter=100,
            outer_step_size=1.0):
        """Optimize the model parameters.
        This is a interface to limetr.
        Args:
            max_iter (int, optional):
                Maximum number of iterations.
            inlier_pct (float, optional):
                How much percentage of the data do you trust.
            outer_max_iter (int, optional):
                Outer maximum number of iterations.
            outer_step_size (float, optional):
                Step size of the trimming problem, the larger the step size the faster it will converge,
                and the less quality of trimming it will guarantee.
        """
        # dimensions for limetr
        n = self.xdata.study_sizes
        if n.size == 0:
            n = np.full(self.xdata.num_obs, 1)
        k_beta = self.num_vars
        k_gamma = 1
        y = self.xdata.obs
        s = self.xdata.obs_se
        x = self.design_mat
        z = np.ones((self.xdata.num_obs, 1))

        uprior = np.hstack((self.prior_beta_uniform, self.prior_gamma_uniform[:, None]))
        gprior = np.hstack((self.prior_beta_gaussian, self.prior_gamma_gaussian[:, None]))

        if self.constraint_mat is None:
            cfun = None
            jcfun = None
            cvec = None
        else:
            num_constraints = self.constraint_mat.shape[0]
            cmat = np.hstack((self.constraint_mat,
                              np.zeros((num_constraints, 1))))

            cvec = np.array([[-np.inf]*num_constraints,
                             [0.0]*num_constraints])

            def cfun(var):
                return cmat.dot(var)

            def jcfun(var):
                return cmat

        def fun(var):
            return x.dot(var)

        def jfun(beta):
            return x

        self.lt = LimeTr(n, k_beta, k_gamma, y, fun, jfun, z,
                         S=s,
                         gprior=gprior,
                         uprior=uprior,
                         C=cfun,
                         JC=jcfun,
                         c=cvec,
                         inlier_percentage=inlier_pct)
        self.beta, self.gamma, self.w = self.lt.fitModel(inner_print_level=5,
                                                         inner_max_iter=max_iter,
                                                         outer_max_iter=outer_max_iter,
                                                         outer_step_size=outer_step_size)

        self.fixed_vars = {
            var: self.beta[self.var_idx[var]]
            for var in self.vars
        }
        if self.use_random_intercept:
            u = self.lt.estimateRE()
            self.random_vars = {
                sid: u[i]
                for i, sid in enumerate(self.xdata.unique_study_id)
            }
        else:
            self.random_vars = dict()

        # compute the posterior distribution of beta
        hessian = self.get_beta_hessian()
        unconstrained_id = np.hstack([
            np.arange(self.lt.k_beta)[self.var_idx[dorm]]
            for dorm in self.xdata.unique_dorms
            if dorm != self.gold_dorm
        ])
        self.beta_sd = np.zeros(self.lt.k_beta)
        self.beta_sd[unconstrained_id] = np.sqrt(np.diag(
            np.linalg.inv(hessian)
        ))

    def get_beta_hessian(self) -> np.ndarray:
        # compute the posterior distribution of beta
        x = self.lt.JF(self.lt.beta)*np.sqrt(self.lt.w)[:, None]
        z = self.lt.Z*np.sqrt(self.lt.w)[:, None]
        v = limetr.utils.VarMat(self.lt.V**self.lt.w, z, self.lt.gamma, self.lt.n)

        if hasattr(self.lt, 'gprior'):
            beta_gprior_sd = self.lt.gprior[:, self.lt.idx_beta][1]
        else:
            beta_gprior_sd = np.repeat(np.inf, self.lt.k_beta)

        hessian = x.T.dot(v.invDot(x)) + np.diag(1.0/beta_gprior_sd**2)
        hessian = np.delete(hessian, self.var_idx[self.gold_dorm], axis=0)
        hessian = np.delete(hessian, self.var_idx[self.gold_dorm], axis=1)

        return hessian

    def get_cov_names(self) -> List[str]:
        # column of covariate name
        cov_names = []
        for model in self.cov_models:
            if model.spline is None:
                cov_names.append(model.alt_cov)
            else:
                cov_names.extend([f'{model.alt_cov}_spline_{i}' for i in range(model.num_x_vars)])
        return cov_names

    def create_result_df(self) -> pd.DataFrame:
        """Create result data frame.

        Returns:
            pd.DataFrame: Data frame that contains the result.
        """
        # column of dorms
        dorms = np.repeat(self.xdata.unique_dorms, self.num_vars_per_dorm)
        cov_names = self.get_cov_names()
        cov_names *= self.xdata.num_dorms

        # create data frame
        df = pd.DataFrame({
            'dorms': dorms,
            'cov_names': cov_names,
            'beta': self.beta,
            'beta_sd': self.beta_sd,
        })
        if self.use_random_intercept:
            gamma = np.hstack((self.lt.gamma, np.full(self.num_vars - 1, np.nan)))
            re = np.hstack((self.lt.u, np.full((self.xdata.num_studies, self.num_vars - 1), np.nan)))
            df['gamma'] = gamma
            for i, study_id in enumerate(self.xdata.unique_study_id):
                df[study_id] = re[i]

        return df

    def save_result_df(self, folder: str, filename: str = 'result.csv'):
        """Save result.

        Args:
            folder (str): Path to the result folder.
            filename (str): Name of the result. Default to `'result.csv'`.
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        df = self.create_result_df()
        df.to_csv(folder + '/' + filename, index=False)
