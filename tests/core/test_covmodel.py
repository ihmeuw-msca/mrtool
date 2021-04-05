"""
Test covariate model
"""
import pytest
import numpy as np
from pandas import DataFrame
from xspline import XSpline
from regmod.utils import SplineSpecs
from mrtool.core.data import MRData
from mrtool.core.cov_model import CovModel, LinearCovModel, LogCovModel
from mrtool.core.prior import GaussianPrior, UniformPrior, SplineGaussianPrior, SplineUniformPrior


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


def test_init_default():
    model = CovModel("cov0")
    assert model.covs == ["cov0"]
    assert model.spline is None
    assert len(model.priors) == 0
    assert len(model.sorted_priors) == 4


@pytest.mark.parametrize("spline", [XSpline([0.0, 1.0], 3),
                                    SplineSpecs([0.0, 1.0], 3)])
def test_spline(spline):
    model = CovModel("cov0", spline=spline)
    assert model.size > 1


@pytest.mark.parametrize("spline", [XSpline([0.0, 1.0], 3),
                                    SplineSpecs([0.0, 1.0], 3)])
def test_attach_data(spline, data):
    model = CovModel("cov0", spline=spline)
    model.attach_data(data)
    assert isinstance(model.spline, XSpline)


def test_prior_init(uprior, gprior, linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=SplineSpecs([0.0, 1.0], 3),
                     priors=[uprior, gprior, linear_gprior, linear_uprior])
    assert len(model.priors) == 4
    assert len(model.sorted_priors["gprior"]) == 1
    assert len(model.sorted_priors["uprior"]) == 1
    assert len(model.sorted_priors["linear_gprior"]) == 1
    assert len(model.sorted_priors["linear_uprior"]) == 1


def test_activate_spline_priors(data, linear_gprior):
    model = CovModel("cov0",
                     spline=SplineSpecs([0.0, 1.0], 3),
                     priors=[linear_gprior])
    assert model.sorted_priors["linear_gprior"][0].mat.size == 0
    model.attach_data(data)
    assert model.sorted_priors["linear_gprior"][0].mat.size > 0


def test_get_uvec(gprior, uprior, linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior, linear_gprior, linear_uprior])
    uvec = model.get_priors("uprior")
    assert np.all(uvec[0] == 0.0)
    assert np.all(uvec[1] == 1.0)


def test_get_uvec_default(linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[linear_gprior, linear_uprior])
    uvec = model.get_priors("uprior")
    assert np.all(np.isinf(uvec[0]))
    assert np.all(np.isinf(uvec[1]))


def test_get_gvec(gprior, uprior, linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior, linear_gprior, linear_uprior])
    gvec = model.get_priors("gprior")
    assert np.all(gvec[0] == 0.0)
    assert np.all(gvec[1] == 1.0)


def test_get_gvec_default(linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[linear_gprior, linear_uprior])
    gvec = model.get_priors("gprior")
    assert np.all(gvec[0] == 0.0)
    assert np.all(np.isinf(gvec[1]))


def test_get_linear_umat(gprior, uprior, linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior, linear_gprior, linear_uprior])
    linear_umat = model.get_priors("linear_uprior")[0]
    assert linear_umat.shape == (linear_uprior.size, model.size)


def test_get_linear_umat_default(gprior, uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior])
    linear_umat = model.get_priors("linear_uprior")[0]
    assert linear_umat.shape == (0, model.size)


def test_get_linear_gmat(gprior, uprior, linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior, linear_gprior, linear_uprior])
    linear_gmat = model.get_priors("linear_gprior")[0]
    assert linear_gmat.shape == (linear_gprior.size, model.size)


def test_get_linear_gmat_default(gprior, uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior])
    linear_gmat = model.get_priors("linear_gprior")[0]
    assert linear_gmat.shape == (0, model.size)


def test_get_linear_uvec(gprior, uprior, linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior, linear_gprior, linear_uprior])
    linear_uvec = model.get_priors("linear_uprior")[1]
    assert linear_uvec.shape == (2, linear_uprior.size)


def test_get_linear_uvec_default(gprior, uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior])
    linear_uvec = model.get_priors("linear_uprior")[1]
    assert linear_uvec.shape == (2, 0)


def test_get_linear_gvec(gprior, uprior, linear_gprior, linear_uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior, linear_gprior, linear_uprior])
    linear_gvec = model.get_priors("linear_gprior")[1]
    assert linear_gvec.shape == (2, linear_uprior.size)


def test_get_linear_gvec_default(gprior, uprior):
    model = CovModel("cov0",
                     spline=XSpline([0.0, 1.0], 3),
                     priors=[gprior, uprior])
    linear_gvec = model.get_priors("linear_gprior")[1]
    assert linear_gvec.shape == (2, 0)


def test_linear_cov_model(data):
    model = LinearCovModel("cov0")
    fun, jac_fun = model.get_fun(data)
    assert np.allclose(fun([1.0]), data["cov0"])
    assert np.allclose(jac_fun([1.0]), model.get_mat(data))


def test_log_cov_model(data):
    model = LogCovModel("cov0")
    fun, jac_fun = model.get_fun(data)
    assert np.allclose(fun([0.0]), 0.0)
    assert np.allclose(jac_fun([0.0]), model.get_mat(data))
