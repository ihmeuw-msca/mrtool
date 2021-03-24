"""
Test DormColumn
"""
import pytest
from pandas import DataFrame
from mrtool.netmr.data import DormColumn, RefDormColumn, AltDormColumn


# pylint:disable=redefined-outer-name


@pytest.fixture
def df():
    return DataFrame({
        "dorm": ["a_b", "b_c", "c_d", "e"]
    })


@pytest.mark.parametrize("separator", [None, "_"])
def test_dormcolumn(df, separator):
    col = DormColumn("dorm", separator=separator)
    col.df = df
    if separator is None:
        assert len(col.unique_values) == 4
        assert all([len(row) == 1 for row in col.values[:3]])
    else:
        assert len(col.unique_values) == 5
        assert all([len(row) == 2 for row in col.values[:3]])


def test_refdormcolumn(df):
    col = RefDormColumn()
    col.df = df
    assert all(df.ref_dorm == "ref_dorm")


def test_altdormcolumn(df):
    col = AltDormColumn()
    col.df = df
    assert all(df.alt_dorm == "alt_dorm")
