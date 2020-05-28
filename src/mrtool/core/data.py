# -*- coding: utf-8 -*-
"""
    data
    ~~~~

    `data` module for `mrtool` package.
"""
from typing import List, Any
from functools import reduce
from operator import and_
import numpy as np
from . import utils


class MRData:
    """Meta-regression data class.
    """
    def __init__(self, df,
                 col_obs=None,
                 col_obs_se=None,
                 col_covs=None,
                 col_study_id=None,
                 add_intercept=False):
        """Constructor of Data
        Args:
            df (pandas.DataFrame):
                Dataframe from csv file that store the data.
            col_obs (str | None, optional):
                Column name that store the observations of the problem.
            col_obs_se (str | None, optional):
                Column name that store the standard error of the observations.
            col_covs (list{str} | None, optional):
                List of column names of potential covariates.
            col_study_id (str | None, optional):
                Column name that store the grouping id of the random effects.
            add_intercept (bool, optional):
                If `True`, add intercept to the current covariates.
        """
        # pass in column names
        self.col_obs = utils.input_cols(col_obs)
        self.col_obs_se = utils.input_cols(col_obs_se)
        self.col_covs = utils.input_cols(col_covs)
        self.col_study_id = utils.input_cols(col_study_id, default='study_id')

        if add_intercept and 'intercept' not in self.cols:
            self.col_covs.append('intercept')

        # pass in data frame
        self.df = None
        self.update_df(df)

    def update_df(self, df):
        """Update the current data frame.
        Args:
            df (pandas.DataFrame):
                New input data frame.
        """
        # add columns if necessary
        if 'intercept' in self.col_covs and 'intercept' not in df:
            df['intercept'] = 1.0
        if self.col_study_id not in df:
            df[self.col_study_id] = np.arange(df.shape[0])

        df = df.sort_values(self.col_study_id).copy()
        self.df = df[self.cols].copy()
        self.df['weights'] = 1.0

    def has_covs(self, covs: List[Any]) -> bool:
        """If the data has the provided covaraites.
        """
        return reduce(and_, [cov in self.col_covs for cov in covs])

    @property
    def cols(self):
        return utils.combine_cols([
            self.__getattribute__(attr)
            for attr in self.__dir__() if 'col_' in attr
        ])

    @property
    def num_obs(self):
        return self.df.shape[0]

    @property
    def num_covs(self):
        return self.covs.shape[1]

    @property
    def num_studies(self):
        return self.study_size.size

    @property
    def obs(self):
        return utils.get_cols(self.df, self.col_obs)

    @property
    def obs_se(self):
        return utils.get_cols(self.df, self.col_obs_se)

    @property
    def covs(self):
        return utils.get_cols(self.df, self.col_covs)

    @property
    def study_id(self):
        return utils.get_cols(self.df, self.col_study_id)

    @property
    def weights(self):
        return utils.get_cols(self.df, 'weights')

    @property
    def study_size(self):
        return self.study_id.value_counts().sort_index()

    def sort_by_study_id(self):
        self.df.sort_values(self.col_study_id, inplace=True)

    def __repr__(self):
        """Summary of the object.
        """
        dimension_summary = [
            "number of observations: %i" % self.num_obs,
            "number of covariates  : %i" % self.num_covs,
            "number of studies     : %i" % self.num_studies,
        ]
        return "\n".join(dimension_summary)
