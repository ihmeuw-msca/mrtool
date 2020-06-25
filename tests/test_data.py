# -*- coding: utf-8 -*-
"""
    test_data
    ~~~~~~~~~
    Test `data` module for `mrtool` package.
"""
import numpy as np
import pandas as pd
import xarray as xr
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

@pytest.fixture()
def xarray():
    example_dataset = xr.Dataset({
        "y":
            xr.DataArray(
                np.random.random([2, 2]),
                dims=["age_group_id", "location_id"],
                name="random_met_need",
                coords={"age_group_id": [2, 3],
                        "location_id": [6, 102]}),
        "y_se":
            xr.DataArray(
                np.ones([2, 2]),
                dims=["age_group_id", "location_id"],
                name="random_met_need",
                coords={"age_group_id": [2, 3],
                        "location_id": [6, 102]}),
        "sdi":
            xr.DataArray(
                np.ones([2, 2])*5,
                dims=["age_group_id", "location_id"],
                name="random_education",
                coords={"age_group_id": [2, 3],
                        "location_id": [6, 102]}),
        "sdi_se":
            xr.DataArray(
                np.ones([2, 2])*0,
                dims=["age_group_id", "location_id"],
                name="random_education",
                coords={"age_group_id": [2, 3],
                        "location_id": [6, 102]}),
    })
    return example_dataset


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


@pytest.mark.parametrize('data_id', [None, 'data_id'])
def test_data_id(df, data_id):
    if data_id is not None:
        df[data_id] = np.arange(df.shape[0])

    d = MRData()
    d.load_df(df,
              col_obs='obs',
              col_obs_se='obs_se',
              col_covs=['cov0', 'cov1', 'cov2'],
              col_data_id=data_id)

    assert (d.data_id == np.arange(d.num_obs)).all()


def test_is_empty(df):
    d = MRData()
    assert d.is_empty()
    d.load_df(df,
              col_obs='obs',
              col_obs_se='obs_se',
              col_covs=['cov0', 'cov1', 'cov2'])
    assert not d.is_empty()
    d.reset()
    assert d.is_empty()


def test_assert_not_empty():
    d = MRData()
    with pytest.raises(ValueError):
        d._assert_not_empty()


def test_has_covs(df):
    d = MRData()
    d.load_df(df,
              col_obs='obs',
              col_obs_se='obs_se',
              col_covs=['cov0', 'cov1', 'cov2'])
    assert d.has_covs(['cov0'])
    assert d.has_covs(['cov0', 'cov1'])
    assert not d.has_covs(['cov3'])

def test_assert_has_covs(df):
    d = MRData()
    d.load_df(df,
              col_obs='obs',
              col_obs_se='obs_se',
              col_covs=['cov0', 'cov1', 'cov2'])
    with pytest.raises(ValueError):
        d._assert_has_covs('cov3')


def test_get_covs(df):
    d = MRData()
    d.load_df(df,
              col_obs='obs',
              col_obs_se='obs_se',
              col_covs=['cov0', 'cov1', 'cov2'])
    for cov_name in ['cov0', 'cov1', 'cov2']:
        assert np.allclose(d.get_covs(cov_name), df[cov_name].to_numpy()[:, None])

    cov_mat = d.get_covs(['cov0', 'cov1', 'cov2'])
    assert np.allclose(cov_mat, df[['cov0', 'cov1', 'cov2']].to_numpy())


@pytest.mark.parametrize('covs', [None, 'cov0', ['cov0', 'cov1']])
def test_normalize_covs(df, covs):
    d = MRData()
    d.load_df(df,
              col_obs='obs',
              col_obs_se='obs_se',
              col_covs=['cov0', 'cov1', 'cov2'])

    d.normalize_covs(covs)
    assert d.is_cov_normalized(covs)


@pytest.mark.parametrize('covs', [['cov0', 'cov1']])
def test_remove_nan_in_covs(df, covs):
    df.loc[:0, covs] = np.nan
    d = MRData()
    with pytest.warns(Warning):
        d.load_df(df,
                  col_obs='obs',
                  col_obs_se='obs_se',
                  col_covs=covs)

    assert d.num_obs == df.shape[0] - 1


def test_load_xr(xarray):
    d = MRData()
    d.load_xr(xarray,
              var_obs='y',
              var_obs_se='y_se',
              var_covs=['sdi'],
              coord_study_id='location_id')

    assert np.allclose(np.sort(d.obs), np.sort(xarray['y'].data.flatten()))
    assert np.allclose(np.sort(d.obs_se), np.sort(xarray['y_se'].data.flatten()))
    assert np.allclose(np.sort(d.covs['sdi']), np.sort(xarray['sdi'].data.flatten()))
    assert np.allclose(np.sort(d.studies), np.sort(xarray.coords['location_id']))