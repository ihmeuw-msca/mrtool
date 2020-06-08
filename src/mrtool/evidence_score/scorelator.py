"""
    scorelator
    ~~~~~~~~~~
"""
from typing import List, Union
import numpy as np
from scipy.integrate import trapz


class Scorelator:
    """Evaluate the score of the result.
    Warning: This is specifically designed for the relative risk application.
    Haven't been tested for others.
    """
    def __init__(self,
                 ln_rr_draws: np.ndarray,
                 exposures: np.ndarray,
                 exposure_domain: Union[List[float], None] = None,
                 ref_exposure: Union[float, None] = None):
        """Constructor of Scorelator.

        Args:
            ln_rr_draws (np.ndarray):
                Draws for log relative risk, each row corresponds to one draw.
            exposures (np.ndarray):
                Exposures pair with the `ln_rr_draws`, its size should match
                the number of columns of `ln_rr_draws`.
            exposure_domain (Union[List[float], None], optional):
                The exposure domain that will be considered for the score evaluation.
                If `None`, consider the whole domain defined by `exposure`.
                Default to None.
            ref_exposure (Union[float, None], optional):
                Reference exposure, the `ln_rr_draws` will be normalized to this point.
                All draws in `ln_rr_draws` will have value 0 at `ref_exposure`.
                If `None`, `ref_exposure` will be set to the minimum value of `exposures`.
                Default to None.
        """
        self.ln_rr_draws = np.array(ln_rr_draws).T
        self.exposures = np.array(exposures)
        self.exposure_domain = [np.min(self.exposures),
                                np.max(self.exposures)] if exposure_domain is None else exposure_domain
        self.ref_exposure = np.min(self.exposures) if ref_exposure is None else ref_exposure
        self._check_inputs()

        self.num_draws = self.ln_rr_draws.shape[0]
        self.num_exposures = self.exposures.size

        # normalize the ln_rr_draws
        self.ln_rr_draws = self.normalize_ln_rr_draws()
        self.rr_draws = np.exp(self.ln_rr_draws)
        self.rr_type = 'harmful' if self.rr_draws[:, -1].mean() >= 1.0 else 'protective'

    def _check_inputs(self):
        """Check the inputs type and value.
        """
        if self.exposures.size != self.ln_rr_draws.shape[1]:
            raise ValueError(f"Size of the exposures ({self.exposures.size}) should be consistent"
                             f"with number of columns of ln_rr_draws ({self.ln_rr_draws.shape[1]})")

        if self.exposure_domain[0] > self.exposure_domain[1]:
            raise ValueError(f"Lower bound of exposure_domain ({self.exposure_domain[0]}) should be"
                             f"less or equal than upper bound ({self.exposure_domain[1]})")

        if not any((self.exposures >= self.exposure_domain[0]) &
                   (self.exposures <= self.exposure_domain[1])):
            raise ValueError("No exposures in the exposure domain.")

        if self.ref_exposure < np.min(self.exposures) or self.ref_exposure > np.max(self.exposures):
            raise ValueError("reference exposure should be within the range of exposures.")

    def normalize_ln_rr_draws(self):
        """Normalize log relative risk draws.
        """
        shift = np.array([
            np.interp(self.ref_exposure, self.exposures, ln_rr_draw)
            for ln_rr_draw in self.ln_rr_draws
        ])
        return self.ln_rr_draws - shift[:, None]

    def get_evidence_score(self,
                           lower_draw_quantile: float = 0.025,
                           upper_draw_quantile: float = 0.975) -> float:
        """Get evidence score.

        Args:
            lower_draw_quantile (float, optional): Lower quantile of the draw for the score.
            upper_draw_quantile (float, optioanl): Upper quantile of the draw for the score.

        Returns:
            float: Evidence score.
        """
        rr_mean = self.rr_draws.mean(axis=0)
        rr_lower = np.quantile(self.rr_draws, lower_draw_quantile, axis=0)
        rr_upper = np.quantile(self.rr_draws, upper_draw_quantile, axis=0)

        valid_index = (self.exposures >= self.exposure_domain[0]) & (self.exposures <= self.exposure_domain[1])

        ab_area = seq_area_between_curves(rr_lower, rr_upper)
        if self.rr_type == 'protective':
            bc_area = seq_area_between_curves(rr_mean, np.ones(self.num_exposures))
            abc_area = seq_area_between_curves(rr_lower, np.ones(self.num_exposures))
        elif self.rr_type == 'harmful':
            bc_area = seq_area_between_curves(np.ones(self.num_exposures), rr_mean)
            abc_area = seq_area_between_curves(np.ones(self.num_exposures), rr_upper)
        else:
            raise ValueError('Unknown relative risk type.')

        score = np.round((ab_area/abc_area)[valid_index[1:]].min(), 2)

        return score


def area_between_curves(lower: np.ndarray,
                        upper: np.ndarray,
                        ind_var: Union[np.ndarray, None] = None,
                        normalize_domain: bool = True) -> float:
    """Compute area between curves.

    Args:
        lower (np.ndarray): Lower bound curve.
        upper (np.ndarray): Upper bound curve.
        ind_var (Union[np.ndarray, None], optional):
            Independent variable, if `None`, it will assume sample points are
            evenly spaced. Default to None.
        normalize_domain (bool, optional):
            If `True`, when `ind_var` is `None`, will normalize domain to 0 and 1.
            Default to True.

    Returns:
        float: Area between curves.
    """
    assert upper.size == lower.size, "Vectors for lower and upper curve should have same size."
    if ind_var is not None:
        assert ind_var.size == lower.size, "Independent variable size should be consistent with the curve vector."
    assert lower.size >= 2, "At least need to have two points to compute interval."

    diff = upper - lower
    if ind_var is not None:
        return trapz(diff, x=ind_var)
    else:
        if normalize_domain:
            dx = 1.0/(diff.size - 1)
        else:
            dx = 1.0
        return trapz(diff, dx=dx)


def seq_area_between_curves(lower: np.ndarray,
                            upper: np.ndarray,
                            ind_var: Union[np.ndarray, None] = None,
                            normalize_domain: bool = True) -> np.ndarray:
    """Sequence areas_between_curves.

    Args: Please check the inputs for area_between_curves.

    Returns:
        np.ndarray:
            Return areas defined from the first two points of the curve
            to the whole curve.
    """
    if ind_var is None:
        return np.array([area_between_curves(
            lower[:(i+1)],
            upper[:(i+1)],
            normalize_domain=normalize_domain)
            for i in range(1, lower.size)
        ])
    else:
        return np.array([area_between_curves(
            lower[:(i + 1)],
            upper[:(i + 1)],
            ind_var[:(i + 1)],
            normalize_domain=normalize_domain)
            for i in range(1, lower.size)
        ])
