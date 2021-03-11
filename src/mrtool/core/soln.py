"""
Solution Module
"""
from typing import Callable, Dict
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray


@dataclass
class MRSoln:

    beta: ndarray
    gamma: ndarray
    beta_vcov: ndarray = field(repr=False)
    gamma_vcov: ndarray = field(repr=False)
    random_effects: Dict

    beta_samples: ndarray = field(init=False, repr=False)
    gamma_samples: ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.beta_samples = np.empty(shape=(0, self.beta.size))
        self.gamma_samples = np.empty(shape=(0, self.gamma.size))

    def sample_soln(self,
                    size: int = 1,
                    sample_beta: bool = True,
                    sample_gamma: bool = False):
        if sample_beta:
            self.beta_samples = np.random.multivariate_normal(
                mean=self.beta,
                cov=self.beta_vcov,
                size=size
            )
        else:
            self.beta_samples = np.repeat(self.beta[None, :], size, axis=0)

        if sample_gamma:
            self.gamma_samples = np.random.multivariate_normal(
                mean=self.gamma,
                cov=self.gamma_vcov,
                size=size
            )
        else:
            self.gamma_samples = np.repeat(self.gamma[None, :], size, axis=0)

    def predict(self,
                fe_fun: Callable,
                re_mat: ndarray,
                group: ndarray = None) -> ndarray:
        fe_pred = fe_fun(self.beta)
        if group is not None:
            re = np.vstack([
                self.random_effects[g] if g in self.random_effects else 0.0
                for g in group
            ])
        else:
            re = np.zeros(re_mat.shape)
        re_pred = (re_mat*re).sum(axis=1)
        return fe_pred + re_pred

    def get_draws(self,
                  fe_fun: Callable,
                  re_mat: ndarray,
                  size: int = 1,
                  sample_beta: bool = True,
                  sample_gamma: bool = False,
                  group: ndarray = None,
                  include_group_uncertainty: bool = True) -> ndarray:

        self.sample_soln(size, sample_beta, sample_gamma)
        fe_pred = np.vstack([fe_fun(beta) for beta in self.beta_samples])
        if include_group_uncertainty:
            re_samples = np.random.randn(size, re_mat.shape[1])*np.sqrt(self.gamma_samples)
            re_pred = np.vstack([re_mat.dot(re) for re in re_samples])
        else:
            if group is not None:
                re = np.vstack([
                    self.random_effects[g] if g in self.random_effects else 0.0
                    for g in group
                ])
            else:
                re = np.zeros(re_mat.shape[1])
            re_pred = (re_mat*re).sum(axis=1)
        return fe_pred + re_pred
