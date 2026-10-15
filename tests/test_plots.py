import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mrtool import LogCovModel, MRBeRT, MRData
from mrtool.core.plots import plot_risk_function
from mrtool.core.utils import sample_knots


@pytest.fixture
def mrbert():
    rng = np.random.default_rng(123)
    num_obs = 30
    alt_lower = rng.uniform(0.1, 0.9, num_obs)
    alt_upper = alt_lower + 0.1
    df = pd.DataFrame(
        {
            "a_0": 0.0,
            "a_1": 0.0,
            "b_0": alt_lower,
            "b_1": alt_upper,
            "obs": np.log(1.0 + 0.5 * (alt_lower + alt_upper))
            + rng.normal(0.0, 0.05, num_obs),
            "obs_se": 0.1,
            "study_id": rng.integers(0, 5, num_obs),
        }
    )
    data = MRData()
    data.load_df(
        df,
        col_obs="obs",
        col_obs_se="obs_se",
        col_covs=["a_0", "a_1", "b_0", "b_1"],
        col_study_id="study_id",
    )
    cov_model = LogCovModel(
        alt_cov=["b_0", "b_1"],
        ref_cov=["a_0", "a_1"],
        use_spline=True,
        spline_knots=np.array([0.0, 0.5, 1.0]),
        spline_knots_type="domain",
        spline_degree=2,
        prior_spline_funval_uniform=[-0.999999, 19.0],
    )
    knots_samples = sample_knots(
        num_knots=1,
        knot_bounds=np.array([[0.3, 0.7]]),
        min_dist=np.array([0.0, 0.0]),
        num_samples=2,
    )
    knots_samples[:, 0] = 0.0
    knots_samples[:, -1] = 1.0
    model = MRBeRT(
        data, ensemble_cov_model=cov_model, ensemble_knots=knots_samples
    )
    model.fit_model()
    return model


def test_plot_risk_function(mrbert, tmp_path):
    np.random.seed(0)
    beta_samples, gamma_samples = mrbert.sample_soln(sample_size=100)
    plot_risk_function(
        mrbert,
        "test_pair",
        beta_samples,
        gamma_samples,
        plots_dir=tmp_path,
        write_file=True,
    )
    assert (tmp_path / "test_pair_risk_function.pdf").is_file()
