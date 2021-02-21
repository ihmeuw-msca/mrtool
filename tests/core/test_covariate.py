"""
Test covariate class
"""
import pytest
import numpy as np
from pandas import DataFrame
from xspline import XSpline
from mrtool.core.data import MRData
from mrtool.core.cov_model import Covariate


# pylint:disable=redefined-outer-name


@pytest.fixture
def df() -> DataFrame:
    np.random.seed(123)
    return DataFrame({
        "obs": np.random.randn(5),
        "obs_se": 0.01 + np.random.rand(5),
        "group": ["A", "A", "B", "C", "D"],
        "key": ["a", "b", "c", "d", "e"],
        "cov0": np.zeros(5) + 0.1*np.random.rand(5),
        "cov1": np.ones(5) + 0.1*np.random.rand(5)
    })


@pytest.fixture
def data(df) -> MRData:
    return MRData.load(
        df=df,
        obs="obs",
        obs_se="obs_se",
        group="group",
        key="key",
        other_cols=["cov0", "cov1"]
    )


@pytest.fixture
def spline() -> XSpline:
    return XSpline(knots=np.linspace(0.0, 1.1, 5), degree=3)


@pytest.mark.parametrize("name", [1, [1, 2]])
def test_init_bad_type(name):
    with pytest.raises(TypeError):
        Covariate(name)


@pytest.mark.parametrize("name", [[], ["a", "b", "c"]])
def test_init_bad_value(name):
    with pytest.raises(ValueError):
        Covariate(name)


@pytest.mark.parametrize("name", ["cov0", ["cov0", "cov1"]])
def test_get_mat(name, data):
    cov = Covariate(name)
    assert np.allclose(cov.get_mat(data), data[cov.name])


@pytest.mark.parametrize("name", ["cov0", ["cov0", "cov1"]])
def test_get_design_mat(name, data):
    cov = Covariate(name)
    assert np.allclose(cov.get_design_mat(data),
                       data[cov.name].mean(axis=1)[:, np.newaxis])


@pytest.mark.parametrize("name", [["cov0", "cov1"]])
def test_get_design_mat_spline(name, data, spline):
    cov = Covariate(name)
    my_mat = cov.get_design_mat(data, spline=spline, use_spline_intercept=True)
    tr_mat = spline.design_imat(data[cov.name[0]],
                                data[cov.name[1]],
                                order=1)/np.diff(data[cov.name], axis=1)
    assert np.allclose(my_mat, tr_mat)
