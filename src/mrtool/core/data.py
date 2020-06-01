# -*- coding: utf-8 -*-
"""
    data
    ~~~~

    `data` module for `mrtool` package.
"""
from typing import Dict, List, Union
from functools import reduce
from operator import and_
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .utils import empty_array


@dataclass
class MRData:
    """Data for simple linear mixed effects model.
    """
    obs: np.ndarray = field(default_factory=empty_array)
    obs_se: np.ndarray = field(default_factory=empty_array)
    covs: Dict[str, np.ndarray] = field(default_factory=dict)
    study_id: np.ndarray = field(default_factory=empty_array)

    def __post_init__(self):
        self._check_attr()
        self.num_obs = self._get_num_obs()
        self._process_attr()
        self.num_covs = len(self.covs)
        self.studies, self.study_sizes = np.unique(self.study_id,
                                                   return_counts=True)
        self.num_studies = len(self.studies)

    def _check_attr(self):
        """Check the type of the attributes.
        """
        assert isinstance(self.obs, np.ndarray)
        assert isinstance(self.obs_se, np.ndarray)
        assert isinstance(self.study_id, np.ndarray)
        assert isinstance(self.covs, dict)
        for cov in self.covs.values():
            assert isinstance(cov, np.ndarray)

    def _get_num_obs(self):
        """Get number of observations.
        """
        num_obs = max([len(self.obs), len(self.obs_se), len(self.study_id)] +
                      [len(cov) for cov in self.covs.values()])
        return num_obs

    def _process_attr(self):
        """Process attribute, including sorting and getting dimensions.
        """
        # add observations
        if len(self.obs) == 0:
            self.obs = np.full(self.num_obs, np.nan)
        else:
            assert len(self.obs) == self.num_obs, "obs, inconsistent size."

        # add obs_se
        if len(self.obs_se) == 0:
            self.obs_se = np.ones(self.num_obs)
        else:
            assert len(self.obs_se) == self.num_obs, "obs_se, inconsistent size."

        # add intercept
        self.covs.update({'intercept': np.ones(self.num_obs)})
        for cov_name in self.covs:
            assert len(self.covs[cov_name]) == self.num_obs, f"covs[{cov_name}], inconsistent size."

        # add study_id
        if len(self.study_id) == 0:
            self.study_id = np.array(['Unknown']*self.num_obs)
        else:
            assert len(self.study_id) == self.num_obs, "study_id, inconsistent size."

        # sort by study_id
        self._sort_by_study_id()

    def _sort_by_study_id(self):
        """Sort data by study_id.
        """
        if self.num_obs != 0 and len(set(self.study_id)) != 1:
            sort_index = np.argsort(self.study_id)
            self.obs = self.obs[sort_index]
            self.obs_se = self.obs_se[sort_index]
            for cov_name in self.covs:
                self.covs[cov_name] = self.covs[cov_name][sort_index]
            self.study_id = self.study_id[sort_index]

    def reset(self):
        """Reset all the attributes to default values.
        """
        self.obs = empty_array()
        self.obs_se = empty_array()
        self.covs = dict()
        self.study_id = empty_array()
        self.__post_init__()

    def load_df(self, df: pd.DataFrame,
                col_obs: Union[str, None] = None,
                col_obs_se: Union[str, None] = None,
                col_covs: Union[List[str], None] = None,
                col_study_id: Union[str, None] = None):
        """Load data from data frame.
        """
        self.reset()

        if col_obs is not None:
            self.obs = df[col_obs].to_numpy()
        else:
            self.obs = empty_array()

        if col_obs_se is not None:
            self.obs_se = df[col_obs_se].to_numpy()
        else:
            self.obs_se = empty_array()

        if col_covs is not None:
            self.covs = {col_cov: df[col_cov].to_numpy()
                         for col_cov in col_covs}
        else:
            self.covs = dict()

        if col_study_id is not None:
            self.study_id = df[col_study_id].to_numpy()
        else:
            self.study_id = empty_array()
        self.__post_init__()

    def to_df(self) -> pd.DataFrame:
        """Convert data object to data frame.
        """
        df = pd.DataFrame({
            'obs': self.obs,
            'obs_se': self.obs_se,
            'study_id': self.study_id
        })
        for cov_name in self.covs:
            df[cov_name] = self.covs[cov_name]

        return df

    def has_covs(self, covs: List[str]) -> bool:
        """If the data has the provided covariates.
        """
        if isinstance(covs, str):
            covs = [covs]
        if len(covs) == 0:
            return True
        else:
            return reduce(and_, [cov in self.covs for cov in covs])

    def get_covs(self, covs: List[str]) -> np.ndarray:
        """Get covariate matrix.
        """
        if isinstance(covs, str):
            covs = [covs]
        assert self.has_covs(covs)
        if len(covs) == 0:
            return np.array([]).reshape(self.num_obs, 0)
        else:
            return np.hstack([self.covs[cov_names][:, None] for cov_names in covs])

    def __repr__(self):
        """Summary of the object.
        """
        dimension_summary = [
            "number of observations: %i" % self.num_obs,
            "number of covariates  : %i" % self.num_covs,
            "number of studies     : %i" % self.num_studies,
        ]
        return "\n".join(dimension_summary)
