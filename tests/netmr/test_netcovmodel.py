"""
Test NetCovModel, currently only linear cov model
"""
import pytest
import numpy as np
import pandas as pd
from mrtool.netmr.data import NetMRData
from mrtool.netmr.cov_model import NetLinearCovModel


# pylint:disable=redefined-outer-name


@pytest.fixture
def df():
    return pd.DataFrame({
        "obs": np.ones(5),
        "obs_se": np.ones(5),
        "study_id": np.arange(5),
        "ref_dorm": ["3_2", "1_2", "2_3", "2_3", "3_1_2"],
        "alt_dorm": ["1_2", "3_2", "1_3", "2", "1"]
    })


@pytest.fixture
def data(df):
    return NetMRData.load(
        df,
        obs="obs",
        obs_se="obs_se",
        group="study_id",
        ref_dorm="ref_dorm",
        alt_dorm="alt_dorm",
        dorm_separator="_"
    )


@pytest.fixture
def cov_model():
    return NetLinearCovModel("intercept")


def test_attach_data(cov_model, data):
    assert cov_model.num_dorms == 1
    cov_model.attach_data(data)
    assert cov_model.num_dorms == 3


def test_size(cov_model, data):
    assert cov_model.size == 1
    cov_model.attach_data(data)
    assert cov_model.size == 3


def test_get_fun(cov_model, data):
    cov_model.attach_data(data)
    _, jac_fun = cov_model.get_fun(data)
    mat = jac_fun(np.zeros(3))
    relation_mat = data.get_relation_mat()
    assert np.allclose(mat, relation_mat)


def test_get_gprior(cov_model, data):
    cov_model.attach_data(data)
    gprior = cov_model.get_priors(ptype="gprior")
    assert gprior.shape[1] == 3
    assert np.allclose(gprior[0], 0.0)
    assert np.all(np.isposinf(gprior[1]))


def test_get_uprior(cov_model, data):
    cov_model.attach_data(data)
    uprior = cov_model.get_priors(ptype="uprior")
    assert uprior.shape[1] == 3
    assert np.all(np.isneginf(uprior[0]))
    assert np.all(np.isposinf(uprior[1]))


def test_get_linear_gprior(cov_model, data):
    cov_model.attach_data(data)
    mat, vec = cov_model.get_priors(ptype="linear_gprior")
    assert mat.shape == (0, 3)
    assert vec.shape == (2, 0)


def test_get_linear_uprior(cov_model, data):
    cov_model.attach_data(data)
    mat, vec = cov_model.get_priors(ptype="linear_uprior")
    assert mat.shape == (0, 3)
    assert vec.shape == (2, 0)
