"""
Test MRSoln Class
"""
import pytest
import numpy as np
from mrtool.core.soln import MRSoln, MRSolnVariable


# pylint:disable=redefined-outer-name


@pytest.fixture
def beta():
    return np.ones(5)


@pytest.fixture
def gamma():
    return np.zeros(3)


@pytest.fixture
def beta_vcov():
    return np.identity(5)


@pytest.fixture
def gamma_vcov():
    return np.zeros((3, 3))


@pytest.fixture
def beta_soln(beta, beta_vcov):
    return MRSolnVariable(beta, beta_vcov)


@pytest.fixture
def gamma_soln(gamma, gamma_vcov):
    return MRSolnVariable(gamma, gamma_vcov)


@pytest.fixture
def random_effects():
    return {group: np.zeros(3) for group in ["a", "b"]}


@pytest.fixture
def soln(beta_soln, gamma_soln, random_effects):
    return MRSoln(beta_soln, gamma_soln, random_effects)


@pytest.fixture
def fe_fun():
    np.random.seed(123)
    fe_mat = np.random.randn(200, 5)

    def fun(beta):
        return fe_mat.dot(beta)
    return fun


@pytest.fixture
def re_mat():
    np.random.seed(123)
    return np.random.randn(200, 3)


def test_soln_variable(beta_soln):
    beta_soln.name = np.array(["a", "b", "c", "c", "d"])
    beta_sub = beta_soln["c"]
    assert beta_sub.size == 2


def test_init(beta_soln, gamma_soln, random_effects):
    soln = MRSoln(beta_soln, gamma_soln, random_effects)
    assert soln.beta_samples.shape == (0, soln.beta.size)
    assert soln.gamma_samples.shape == (0, soln.gamma.size)


@pytest.mark.parametrize("size", [100])
@pytest.mark.parametrize("sample_beta", [True, False])
@pytest.mark.parametrize("sample_gamma", [True, False])
def test_sample_soln(soln, size, sample_beta, sample_gamma):
    soln.sample_soln(size, sample_beta=sample_beta, sample_gamma=sample_gamma)
    assert soln.beta_samples.shape == (size, soln.beta.size)
    assert soln.gamma_samples.shape == (size, soln.gamma.size)


@pytest.mark.parametrize("group", [np.array(["a"]*200)])
def test_predict(soln, fe_fun, re_mat, group):
    pred = soln.predict(fe_fun, re_mat, group)
    assert pred.size == 200


@pytest.mark.parametrize("size", [100])
@pytest.mark.parametrize("sample_beta", [True, False])
@pytest.mark.parametrize("sample_gamma", [True, False])
@pytest.mark.parametrize("group", [np.array(["uknown"]*200)])
@pytest.mark.parametrize("include_group_uncertainty", [True, False])
def test_get_draws(soln, fe_fun, re_mat,
                   size, sample_beta, sample_gamma,
                   group, include_group_uncertainty):
    draws = soln.get_draws(fe_fun, re_mat, group,
                           size, sample_beta, sample_gamma,
                           include_group_uncertainty)
    assert draws.shape == (size, 200)
