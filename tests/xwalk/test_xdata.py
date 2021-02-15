# -*- coding: utf-8 -*-
"""
    test_data
    ~~~~~~~~~
    Test `data` model for `crosswalk` package.
"""
import numpy as np
import pandas as pd
import pytest
from mrtool.xwalk.data import XData


# test case settings
num_obs = 5
num_covs = 3


@pytest.fixture()
def df():
    df = pd.DataFrame({
        'obs': np.random.randn(num_obs),
        'obs_se': np.random.rand(num_obs) + 0.01,
        'alt_dorms': np.arange(num_obs),
        'ref_dorms': np.arange(num_obs)[::-1],
        'study_id': np.array([2, 1, 2, 1, 3]),
        'data_id': np.array(['A', 'B', 'C', 'D', 'E'])
    })
    for i in range(num_covs):
        df['cov%i' % i] = np.random.randn(num_obs)
    return df


@pytest.mark.parametrize("study_id", [None, "study_id"])
def test_xdata_study_id(df, study_id):
    xdata = XData()
    xdata.load_df(df,
                  col_obs='obs',
                  col_obs_se='obs_se',
                  col_alt_dorms='alt_dorms',
                  col_ref_dorms='ref_dorms',
                  col_covs=['cov%i' % i for i in range(num_covs)],
                  col_study_id=study_id)

    if study_id is not None:
        assert xdata.num_studies == 3
        assert tuple(xdata.study_sizes) == (2, 2, 1)
        assert tuple(xdata.studies) == (1, 2, 3)
        assert tuple(xdata.study_id) == (1, 1, 2, 2, 3)
    else:
        assert xdata.num_studies == 1
        assert xdata.study_sizes.size == 1
        assert xdata.studies.size == 1


@pytest.mark.parametrize('data_id', [None, 'data_id'])
def test_data_id(df, data_id):
    xdata = XData()
    xdata.load_df(df,
                  col_obs='obs',
                  col_obs_se='obs_se',
                  col_alt_dorms='alt_dorms',
                  col_ref_dorms='ref_dorms',
                  col_covs=['cov%i' % i for i in range(num_covs)],
                  col_data_id=data_id)

    if data_id is None:
        assert (np.sort(xdata.data_id) == np.arange(xdata.num_obs)).all()
    else:
        assert (np.sort(xdata.data_id) == np.sort(df[data_id].to_numpy())).all()
