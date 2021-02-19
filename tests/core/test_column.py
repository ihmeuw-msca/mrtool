"""
Test column classes
"""
import pytest
import numpy as np
from pandas import DataFrame
from mrtool.core.data import Column, SEColumn, GroupColumn, KeyColumn, InterceptColumn

# pylint:disable=redefined-outer-name


@pytest.fixture
def df() -> DataFrame:
    np.random.seed(123)
    return DataFrame({
        "obs": np.random.randn(5),
        "obs_se": 0.01 + np.random.rand(5),
        "group": ["A", "A", "B", "C", "D"],
        "key": ["a", "b", "c", "d", "e"],
        "cov": np.random.randn(5)
    })


def test_col_init_default():
    col = Column()
    assert col.name is None


@pytest.mark.parametrize("name", ["name"])
def test_col_init(name):
    col = Column(name)
    assert col.name == name


def test_col_df_default(df):
    col = Column()
    col.df = df


@pytest.mark.parametrize("name", ["obs"])
def test_col_df_good(df, name):
    col = Column(name)
    col.df = df


@pytest.mark.parametrize("name", ["name"])
def test_col_df_bad(df, name):
    col = Column(name)
    with pytest.raises(ValueError):
        col.df = df
    df[name] = np.nan
    with pytest.raises(ValueError):
        col.df = df


def test_col_is_empty(df):
    col = Column()
    assert col.is_empty
    col.df = df
    assert col.is_empty
    col.df = None
    col.name = "obs"
    assert col.is_empty
    col.df = df
    assert not col.is_empty


@pytest.mark.parametrize("name", ["obs"])
def test_col_values_good(df, name):
    col = Column(name)
    col.df = df
    assert np.allclose(col.values, df[name])


@pytest.mark.parametrize("name", ["obs"])
def test_col_values_bad(df, name):
    col = Column(name)
    with pytest.raises(ValueError):
        assert np.allclose(col.values, df[name])


@pytest.mark.parametrize("name", ["group"])
def test_col_unique_values(df, name):
    col = Column(name)
    col.df = df
    unique_values = set(col.unique_values)
    assert unique_values == set(["A", "B", "C", "D"])


@pytest.mark.parametrize("name", ["group"])
def test_col_value_counts(df, name):
    col = Column(name)
    col.df = df
    value_counts = col.value_counts
    assert value_counts["A"] == 2
    assert value_counts["B"] == 1
    assert value_counts["C"] == 1
    assert value_counts["D"] == 1


def test_secol_init_default():
    col = SEColumn()
    assert col.name == "obs_se"


@pytest.mark.parametrize("name", ["obs_se1"])
def test_secol_df_value_default(df, name):
    col = SEColumn(name=name)
    col.df = df
    assert np.allclose(col.values, 1)


@pytest.mark.parametrize("name", ["obs_se1"])
def test_secol_df_value_bad(df, name):
    col = SEColumn(name=name)
    df[name] = -1
    with pytest.raises(ValueError):
        col.df = df


def test_keycol_init_default():
    col = KeyColumn()
    assert col.name == "key"


@pytest.mark.parametrize("name", ["key1"])
def test_keycol_df_value_default(df, name):
    col = KeyColumn(name=name)
    col.df = df
    assert np.allclose(col.values, np.arange(df.shape[0]))


@pytest.mark.parametrize("name", ["key1"])
def test_keycol_df_value_bad(df, name):
    col = KeyColumn(name=name)
    df[name] = 1
    with pytest.raises(ValueError):
        col.df = df


def test_groupcol_init_default():
    col = GroupColumn()
    assert col.name == "group"


@pytest.mark.parametrize("name", ["group1"])
def test_groupcol_df_value_default(df, name):
    col = GroupColumn(name=name)
    col.df = df
    assert np.all(col.values == "unknown")


def test_interceptcol_init_default():
    col = GroupColumn()
    assert col.name == "group"


@pytest.mark.parametrize("name", ["intercept1"])
def test_interceptcol_df_value_default(df, name):
    col = InterceptColumn(name=name)
    col.df = df
    assert np.allclose(col.values, 1)


@pytest.mark.parametrize("name", ["intercept1"])
def test_interceptcol_df_value_bad(df, name):
    col = InterceptColumn(name=name)
    df[name] = 2
    with pytest.raises(ValueError):
        col.df = df
