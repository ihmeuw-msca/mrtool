# -*- coding: utf-8 -*-
"""
    Cov Finder
    ~~~~~~~~~~
"""
from typing import List, Tuple, Union
from copy import deepcopy
import numpy as np
from mrtool import MRData, LinearCovModel, MRBRT

class CovFinder:
    """Class in charge of the covariate selection.
    """
    loose_beta_gprior_std = 100.0
    dummy_gamma_uprior = np.array([100.0, 100.0])

    def __init__(self,
                 data: MRData,
                 covs: List[str],
                 pre_selected_covs: Union[List[str], None] = None,
                 normalized_covs: bool = True,
                 num_samples: int = 1000,
                 laplace_threshold: float = 1e-5,
                 power_range: Tuple[float, float] = (-8, 8),
                 power_step_size: float = 0.5):
        """Covariate Finder.

        Args:
            data (MRData): Data object used for variable selection.
            covs (List[str]): Candidate covariates.
            normalized_covs (bool): If true, will normalize the covariates.
            pre_selected_covs (List[str] | None, optional):
                Pre-selected covaraites, will always be in the selected list.
            num_samples (int, optional):
                Number of samples used for judging if a variable is significance.
            laplace_threshold (float, optional):
                When coefficients from the Laplace regression is above this value,
                we consider it as the potential useful covariate.
            power_range (Tuple[float, float], optional):
                Power range for the Laplace prior standard deviation.
                Laplace prior standard deviation will go from `10**power_range[0]`
                to `10**power_range[1]`.
            power_step_size (float, optional): Step size of the swiping across the power range.
        """

        self.data = data
        self.covs = covs
        self.pre_selected_covs = [] if pre_selected_covs is None else pre_selected_covs
        assert len(set(self.pre_selected_covs) & set(self.covs)) == 0, \
            "covs and pre_selected_covs should be mutually exclusive."
        self.normalize_covs = normalized_covs
        if self.normalize_covs:
            self.data = deepcopy(data)
            self.data.normalize_covs(self.covs)
        self.selected_covs = self.pre_selected_covs.copy()
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
                               prior_beta_laplace=np.array([0.0, laplace_std])
                               if cov not in self.selected_covs else None,
                               prior_beta_gaussian=None
                               if cov not in self.pre_selected_covs else self.beta_gprior[cov],
                               prior_gamma_uniform=self.dummy_gamma_uprior)
                for cov in covs
            ]
        else:
            cov_models = [
                LinearCovModel(cov, use_re=True,
                               prior_beta_gaussian=np.array([0.0, self.loose_beta_gprior_std])
                               if cov not in self.beta_gprior else self.beta_gprior[cov],
                               prior_gamma_uniform=self.dummy_gamma_uprior)
                for cov in covs
            ]

        return MRBRT(self.data, cov_models=cov_models)

    def select_covs_by_laplace(self, laplace_std: float, verbose: bool = False):
        # laplace model
        laplace_model = self.create_model(self.covs,
                                          prior_type='Laplace',
                                          laplace_std=laplace_std)

        laplace_model.fit_model(x0=np.zeros(2*laplace_model.num_vars))
        additional_covs = [
            cov
            for i, cov in enumerate(self.covs)
            if cov not in self.selected_covs and laplace_model.beta_soln[i] > self.laplace_threshold
        ]
        if verbose:
            print('potential additional covariates', additional_covs)

        if len(additional_covs) > 0:
            candidate_covs = self.selected_covs + additional_covs
            candidate_cov_dict = {
                cov: i for i, cov in enumerate(candidate_covs)
            }
            gaussian_model = self.create_model(candidate_covs,
                                               prior_type='Gaussian')
            gaussian_model.fit_model(x0=np.zeros(gaussian_model.num_vars))
            beta_soln_samples, _ = gaussian_model.sample_soln(sample_size=self.num_samples,
                                                              sim_prior=False,
                                                              sim_re=False,
                                                              print_level=5)
            beta_soln_mean = gaussian_model.beta_soln
            beta_soln_std = np.std(beta_soln_samples, axis=0)
            beta_soln_sig = self.is_significance(beta_soln_samples, var_type='beta')
            if verbose:
                print('    mean:', beta_soln_mean)
                print('    std:', beta_soln_std)
                print('    significance:', beta_soln_sig)
            # update the selected covs
            self.selected_covs.extend([
                cov for i, cov in enumerate(candidate_covs)
                if cov not in self.selected_covs and beta_soln_sig[i]
            ])
            if verbose:
                print('    selected covaraites:', self.selected_covs)
            # update beta_gprior
            self.beta_gprior.update({
                cov: np.array([beta_soln_mean[candidate_cov_dict[cov]], self.loose_beta_gprior_std])
                for cov in self.selected_covs
                if cov not in self.beta_gprior
            })
            # update the stop
            self.stop = not all(beta_soln_sig)

        # other stop criterion
        if len(self.selected_covs) == self.num_covs:
            self.stop = True

    def select_covs(self, verbose: bool = False):
        for power in self.powers:
            if not self.stop:
                laplace_std = 10**(0.5*power)
                self.select_covs_by_laplace(laplace_std, verbose=verbose)
        self.stop = True

    @staticmethod
    def is_significance(var_samples: np.ndarray,
                        var_type: str = 'beta') -> List[bool]:
        assert var_type == 'beta', "Only support variable type beta."
        var_uis = np.percentile(var_samples, (2.5, 97.5), axis=0)
        var_sig = var_uis.prod(axis=0) > 0

        return var_sig
