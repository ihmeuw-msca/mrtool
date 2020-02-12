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
    d = MRData(df,
               col_obs=obs,
               col_obs_se=obs_se,
               col_covs=['cov0', 'cov1', 'cov2'])
    if obs is None:
        assert d.obs.shape == (df.shape[0], 0)
    else:
        assert d.obs.shape == (df.shape[0],)

    if obs_se is None:
        assert d.obs_se.shape == (df.shape[0], 0)
    else:
        assert d.obs_se.shape == (df.shape[0],)

    assert d.num_obs == df.shape[0]


@pytest.mark.parametrize('covs', [None,
                                  ['cov0', 'cov1', 'cov2']])
@pytest.mark.parametrize('add_intercept', [True, False])
def test_covs(df, covs, add_intercept):
    d = MRData(df,
               col_obs='obs',
               col_obs_se='obs_se',
               col_covs=covs,
               add_intercept=add_intercept)

    num_covs = 0 if covs is None else len(covs)
    num_covs += add_intercept
    print(d.col_covs)
    assert d.covs.shape == (d.num_obs, num_covs)


@pytest.mark.parametrize('study_id', [None, np.array([0, 0, 1, 1, 2])])
def test_study_id(df, study_id):
    if study_id is not None:
        df['study_id'] = study_id
        col_study_id = 'study_id'
    else:
        col_study_id = None
    d = MRData(df,
               col_obs='obs',
               col_obs_se='obs_se',
               col_covs=['cov0', 'cov1', 'cov2'],
               col_study_id=col_study_id)

    if col_study_id is None:
        assert np.allclose(d.study_id, np.arange(d.num_obs))
        assert d.num_studies == d.num_obs
        assert np.allclose(d.study_size.index, np.arange(d.num_obs))
        assert np.allclose(d.study_size, 1.0)
    else:
        assert np.allclose(d.study_id, np.array([0, 0, 1, 1, 2]))
        assert d.num_studies == 3
        assert np.allclose(d.study_size.index, np.array([0, 1, 2]))
        assert np.allclose(d.study_size, np.array([2, 2, 1]))


@pytest.mark.parametrize('study_id', [np.arange(5)])
def test_sort_by(df, study_id):
    df['study_id'] = study_id
    d = MRData(df,
               col_obs='obs',
               col_obs_se='obs_se',
               col_covs=['cov0', 'cov1', 'cov2'],
               col_study_id='study_id')

    d.sort_by_study_id()
    assert np.allclose(d.study_id, np.arange(d.num_obs))
