"""
Solution Module
"""
from typing import Callable, Dict, Iterable, List
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray


@dataclass
class MRSolnVariable:

    mean: Iterable[float]
    vcov: Iterable[float]
    name: List[str] = None

    def __post_init__(self):
        self.mean = np.asarray(self.mean)
        self.vcov = np.asarray(self.vcov)
        if self.name is None:
            self.name = np.tile(None, self.size)
        else:
            self.name = np.asarray(self.name)

        if self.vcov.shape != (self.size, self.size):
            raise ValueError(f"vcov must be shape {(self.size, self.size)}.")

        if self.name.size != self.size:
            raise ValueError(f"name must be size{self.size}")

    @property
    def size(self) -> int:
        return self.mean.size

    def get_index(self, key: str) -> ndarray:
        return self.name == key

    def __getitem__(self, key: str):
        return self.mean[self.get_index(key)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


@dataclass
class MRSoln:

    beta: MRSolnVariable
    gamma: MRSolnVariable
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
                mean=self.beta.mean,
                cov=self.beta.vcov,
                size=size
            )
        else:
            self.beta_samples = np.repeat(self.beta.mean[None, :], size, axis=0)

        if sample_gamma:
            self.gamma_samples = np.random.multivariate_normal(
                mean=self.gamma.mean,
                cov=self.gamma.vcov,
                size=size
            )
        else:
            self.gamma_samples = np.repeat(self.gamma.mean[None, :], size, axis=0)

    def get_random_effects(self, group: ndarray) -> ndarray:
        return np.vstack([
            self.random_effects[g] if g in self.random_effects else 0.0
            for g in group
        ])

    def predict(self,
                fe_fun: Callable,
                re_mat: ndarray,
                group: ndarray) -> ndarray:
        fe_pred = fe_fun(self.beta.mean)
        re = self.get_random_effects(group)
        re_pred = (re_mat*re).sum(axis=1)
        return fe_pred + re_pred

    def get_draws(self,
                  fe_fun: Callable,
                  re_mat: ndarray,
                  group: ndarray,
                  size: int = 1,
                  sample_beta: bool = True,
                  sample_gamma: bool = False,
                  include_group_uncertainty: bool = True) -> ndarray:

        if size == 0:
            return np.empty(shape=(0, re_mat.shape[0]))

        self.sample_soln(size, sample_beta, sample_gamma)
        fe_pred = np.vstack([fe_fun(beta) for beta in self.beta_samples])
        if include_group_uncertainty:
            re_samples = np.random.randn(size, re_mat.shape[1])*np.sqrt(self.gamma_samples)
            re_pred = np.vstack([re_mat.dot(re) for re in re_samples])
        else:
            re = self.get_random_effects(group)
            re_pred = (re_mat*re).sum(axis=1)
        return fe_pred + re_pred
