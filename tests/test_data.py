# -*- coding: utf-8 -*-
"""
    test_data
    ~~~~~~~~~
    Test `data` module for `mrtool` package.
"""
import numpy as np
import pandas as pd
import pytest
from mrtool import MRData


@pytest.fixture()
def df():
    num_obs = 5
    df = pd.DataFrame({
        'obs': np.random.randn(num_obs),
        'obs_se': np.random.rand(num_obs) + 0.01,
        'cov0': np.random.randn(num_obs),
        'cov1': np.random.randn(num_obs),
        'cov2': np.random.randn(num_obs),
    })
    return df


@pytest.mark.parametrize('obs', ['obs', None])
@pytest.mark.parametrize('obs_se', ['obs_se', None])
def test_obs(df, obs, obs_se):
    d = MRData()
    d.load_df(df,
              col_obs=obs,
              col_obs_se=obs_se,
              col_covs=['cov0', 'cov1', 'cov2'])
    assert d.obs.size == df.shape[0]
    assert d.obs_se.size == df.shape[0]
    if obs is None:
        assert all(np.isnan(d.obs))


@pytest.mark.parametrize('covs', [None,
                                  ['cov0', 'cov1', 'cov2']])
def test_covs(df, covs):
    d = MRData()
    d.load_df(df,
              col_obs='obs',
              col_obs_se='obs_se',
              col_covs=covs)

    num_covs = 0 if covs is None else len(covs)
    num_covs += 1
    assert d.num_covs == num_covs


@pytest.mark.parametrize('study_id', [None, np.array([0, 0, 1, 1, 2])])
def test_study_id(df, study_id):
    if study_id is not None:
        df['study_id'] = study_id
        col_study_id = 'study_id'
    else:
        col_study_id = None
    d = MRData()
    d.load_df(df,
              col_obs='obs',
              col_obs_se='obs_se',
              col_covs=['cov0', 'cov1', 'cov2'],
              col_study_id=col_study_id)

    if col_study_id is None:
        assert np.all(d.study_id == 'Unknown')
        assert d.num_studies == 1
        assert d.studies[0] == 'Unknown'
    else:
        assert np.allclose(d.study_id, np.array([0, 0, 1, 1, 2]))
        assert d.num_studies == 3
        assert np.allclose(d.studies, np.array([0, 1, 2]))
        assert np.allclose(d.study_sizes, np.array([2, 2, 1]))
