# -*- coding: utf-8 -*-
"""
    Cov Finder
    ~~~~~~~~~~
"""
from typing import List, Tuple
import numpy as np
from mrtool import MRData, LinearCovModel, MRBRT

class CovFinder:
    """Class in charge of the covariate selection.
    """
    loose_beta_gprior_std = 0.1
    dummy_gamma_uprior = [1000.0, 1000.0]

    def __init__(self,
                 data: MRData,
                 covs: List[str],
                 num_samples: int = 1000,
                 laplace_threshold: float = 1e-5,
                 power_range: Tuple[float, float] = (-8, 8),
                 power_step_size: float = 0.5):

        self.data = data
        self.covs = covs
        self.selected_covs = []
        self.beta_gprior = {}
        self.stop = False

        self.num_samples = num_samples
        self.laplace_threshold = laplace_threshold
        self.power_range = power_range
        self.power_step_size = power_step_size
        self.powers = np.arange(*self.power_range, self.power_step_size)

        self.num_covs = len(covs)

    def create_model(self,
                     covs: List[str],
                     prior_type: str = 'Laplace',
                     laplace_std: float = None) -> MRBRT:
        assert prior_type in ['Laplace', 'Gaussian'], "Prior type can only 'Laplace' or 'Gaussian'."
        if prior_type == 'Laplace':
            assert laplace_std is not None, "Use Laplace prior must provide standard deviation."

        if prior_type == 'Laplace':
            cov_models = [
                LinearCovModel(cov, use_re=True,
                               prior_beta_laplace=[0.0, laplace_std],
                               prior_gamma_uniform=self.dummy_gamma_uprior)
                for cov in covs
            ]
        else:
            cov_models = [
                LinearCovModel(cov, use_re=True,
                               prior_beta_gaussian=[0.0, self.loose_beta_gprior_std]
                               if cov not in self.beta_gprior else self.beta_gprior[cov],
                               prior_gamma_uniform=self.dummy_gamma_uprior)
                for cov in covs
            ]

        return MRBRT(self.data, cov_models=cov_models)

    def select_covs_by_laplace(self, laplace_std: float):
        # laplace model
        laplace_model = self.create_model(self.covs,
                                          prior_type='Laplace',
                                          laplace_std=laplace_std)

        laplace_model.fit_model()
        additional_covs = [
            cov
            for i, cov in enumerate(self.covs)
            if cov not in self.selected_covs and laplace_model.beta_soln[i] > self.laplace_threshold
        ]

        if len(additional_covs) > 0:
            candidate_covs = self.selected_covs + additional_covs
            gaussian_model = self.create_model(candidate_covs,
                                               prior_type='Gaussian')
            gaussian_model.fit_model()
            beta_soln_samples, _ = gaussian_model.sample_soln(sample_size=self.num_samples)
            beta_soln_mean = gaussian_model.beta_soln
            beta_soln_std = np.std(beta_soln_samples, axis=0)
            beta_soln_sig = self.is_significance(beta_soln_samples, var_type='beta')
            # update the selected covs
            self.selected_covs.extend([
                cov for i, cov in enumerate(candidate_covs)
                if cov not in self.selected_covs and beta_soln_sig[i]
            ])
            # update beta_gprior
            self.beta_gprior.update({
                cov: [beta_soln_mean[i], beta_soln_std[i]]
                for i, cov in enumerate(self.selected_covs)
                if cov not in self.beta_gprior
            })
            # update the stop
            self.stop = not all(beta_soln_sig)

        # other stop criterion
        if len(self.selected_covs) == self.num_covs:
            self.stop = True

    def select_covs(self):
        for power in self.powers:
            if not self.stop:
                laplace_std = 10**(0.5*power)
                self.select_covs_by_laplace(laplace_std)

    @staticmethod
    def is_significance(var_samples: np.ndarray,
                        var_type: str = 'beta') -> List[bool]:
        assert var_type == 'beta', "Only support variable type beta."
        var_uis = np.percentile(var_samples, (2.5, 97.5), axis=0)
        var_sig = var_uis.prod(axis=0) > 0

        return var_sig
