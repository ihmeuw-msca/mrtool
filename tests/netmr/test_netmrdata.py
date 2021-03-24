"""
Test NetMRData
"""
import pytest
import numpy as np
from pandas import DataFrame
from mrtool.netmr.data import NetMRData


# pylint:disable=redefined-outer-name


@pytest.fixture
def df():
    np.random.seed(123)
    return DataFrame({
        "obs": np.random.randn(5),
        "obs_se": np.random.rand(5),
        "group": ["A", "A", "B", "B", "C"],
        "ref_dorm": ["1_2", "3", "3", "3", "3"],
        "alt_dorm": ["3", "2", "1", "2", "1"]
    })


@pytest.fixture
def data():
    return NetMRData(
        obs="obs",
        obs_se="obs_se",
        group="study_id",
        ref_dorm="ref_dorm",
        alt_dorm="alt_dorm",
        dorm_separator="_"
    )


def test_data_empty(data):
    assert data.is_empty


def test_data(data, df):
    data.df = df
    assert len(data.ref_dorm.unique_values) == 3
    assert len(data.alt_dorm.unique_values) == 3
    assert len(data.unique_dorms) == 3
