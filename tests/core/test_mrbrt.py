"""
Test MRBRT Class
"""
import pytest
import numpy as np
from pandas import DataFrame
from regmod.utils import SplineSpecs
from mrtool.core.data import MRData
from mrtool.core.cov_model import LinearCovModel
from mrtool.core.prior import GaussianPrior, UniformPrior, SplineGaussianPrior, SplineUniformPrior
from mrtool.core.model import MRBRT


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
def gprior():
    return GaussianPrior(mean=0.0, sd=1.0)


@pytest.fixture
def uprior():
    return UniformPrior(lb=0.0, ub=1.0)


@pytest.fixture
def linear_gprior():
    return SplineGaussianPrior(size=100, mean=0.0, sd=1.0, order=1)


@pytest.fixture
def linear_uprior():
    return SplineUniformPrior(size=100, lb=0.0, ub=1.0, order=1)


@pytest.fixture
def spline():
    return SplineSpecs([0.0, 1.0], 3)


@pytest.fixture
def cov_model(spline, uprior, gprior, linear_uprior, linear_gprior):
    return LinearCovModel("cov0", spline=spline,
                          priors=[uprior, gprior, linear_uprior, linear_gprior])


@pytest.fixture
def model(data, cov_model):
    return MRBRT(data, cov_model)


@pytest.mark.parametrize("data", [1.0])
def test_data_check(data, cov_model):
    with pytest.raises(TypeError):
        MRBRT(data, cov_model)


@pytest.mark.parametrize("cov_model", [1.0])
def test_cov_model_check(data, cov_model):
    with pytest.raises(TypeError):
        MRBRT(data, cov_model)


@pytest.mark.parametrize("pct", [-1.0, 2.0])
def test_inlier_pct_check(data, cov_model, pct):
    with pytest.raises(ValueError):
        MRBRT(data, cov_model, inlier_pct=pct)


def test_fe_sizes(model):
    assert model.fe_sizes == [model.fe_cov_models[0].size]


def test_re_sizes(model):
    assert model.re_sizes == []


def test_fe_size(model):
    assert model.fe_size == model.fe_cov_models[0].size


def test_re_size(model):
    assert model.re_size == 0


def test_size(model):
    assert model.size == model.fe_size + model.re_size


def test_uvec(model, uprior):
    assert np.allclose(model.uvec[0], uprior.lb)
    assert np.allclose(model.uvec[1], uprior.ub)


def test_gvec(model, gprior):
    assert np.allclose(model.gvec[0], gprior.mean)
    assert np.allclose(model.gvec[1], gprior.sd)


def test_linear_umat(model, linear_uprior):
    assert np.allclose(model.linear_umat, linear_uprior.mat)


def test_linear_uvec(model, linear_uprior):
    assert np.allclose(model.linear_uvec[0], linear_uprior.lb)
    assert np.allclose(model.linear_uvec[1], linear_uprior.ub)


def test_linear_gmat(model, linear_gprior):
    assert np.allclose(model.linear_gmat, linear_gprior.mat)


def test_linear_gvec(model, linear_gprior):
    assert np.allclose(model.linear_gvec[0], linear_gprior.mean)
    assert np.allclose(model.linear_gvec[1], linear_gprior.sd)
