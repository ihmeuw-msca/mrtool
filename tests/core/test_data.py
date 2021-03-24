"""
Test data module
"""
import pytest
import numpy as np
from pandas import DataFrame
from mrtool.core.data import Column, MRData

# pylint:disable=redefined-outer-name


@pytest.fixture
def df() -> DataFrame:
    np.random.seed(123)
    return DataFrame({
        "obs": np.random.randn(5),
        "obs_se": 0.01 + np.random.rand(5),
        "group": ["A", "A", "B", "C", "D"],
        "key": ["a", "b", "c", "d", "e"],
        "cov": np.random.randn(5),
        "cov_other": np.random.randn(5)
    })


@pytest.fixture
def data() -> MRData:
    return MRData(
        obs="obs",
        obs_se="obs_se",
        group="group",
        key="key",
        other_cols=["cov"]
    )


def test_init_default():
    data = MRData()
    assert data.df is None
    assert "intercept" in data.col_names


def test_init(data):
    assert all([col in data.col_names
                for col in ["obs", "obs_se", "group", "key", "cov", "intercept"]])


def test_df_good(df, data):
    data.df = df
    assert np.allclose(data.obs.values, data.df[data.obs.name])


def test_df_bad(df, data):
    df[data.obs.name] = np.nan
    with pytest.raises(ValueError):
        data.df = df
    df = df[[col_name for col_name in data.col_names
             if col_name not in [data.obs.name, "intercept"]]]
    with pytest.raises(ValueError):
        data.df = df


def test_is_empty(df, data):
    assert data.is_empty
    data.df = df
    assert not data.is_empty


def test_shape(df, data):
    data.df = df
    assert data.shape == df.shape


def test_getitem(df, data):
    data.df = df
    assert np.allclose(data[data.obs.name], df[data.obs.name])


def test_sort_values(df, data):
    data.df = df
    data.sort_values(data.obs.name)
    assert all(np.diff(data.obs.values) >= 0)


def test_load(df):
    data = MRData.load(
        df,
        obs="obs",
        obs_se="obs_se",
        group="group",
        key="key",
        other_cols=["cov"]
    )
    assert data.shape == df.shape


def test_copy(df, data):
    data.df = df
    new_data = data.copy()
    assert set(data.col_names) == set(new_data.col_names)
    assert new_data.is_empty


def test_add_column(df, data):
    data.df = df
    col_names = data.col_names.copy()
    data.add_column(Column("cov_other"))
    new_col_names = data.col_names.copy()

    assert "cov_other" in new_col_names
    assert "cov_other" not in col_names
