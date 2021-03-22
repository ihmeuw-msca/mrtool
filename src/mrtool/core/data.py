# -*- coding: utf-8 -*-
"""
    data
    ~~~~

    `data` module for `mrtool` package.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame


@dataclass
class Column:
    """
    General DataFrame column
    """

    name: str = None
    _df: DataFrame = field(default=None, init=False, repr=False)

    @staticmethod
    def _check_df_type(df: DataFrame):
        if not (isinstance(df, DataFrame) or df is None):
            raise TypeError("df must be DataFrame or None.")

    def _check_df_values(self, df: DataFrame):
        if self.name not in df.columns:
            self._set_default_values(df)
        if any(df[self.name].isna()):
            raise ValueError(f"Column({self.name}) values cannot contain NaN.")

    def _set_default_values(self, df: DataFrame):
        raise ValueError(f"df must contain column {self.name}.")

    def _check_df(self, df: DataFrame):
        self._check_df_type(df)
        if (df is not None) and (self.name is not None):
            self._check_df_values(df)

    def _assert_not_empty(self):
        if self.is_empty:
            raise ValueError("Please attach df and/or specify col name.")

    @property
    def df(self) -> DataFrame:
        return self._df

    @df.setter
    def df(self, df: DataFrame):
        self._check_df(df)
        self._df = df

    @property
    def is_empty(self) -> bool:
        return self.df is None or self.name is None

    @property
    def values(self) -> ndarray:
        self._assert_not_empty()
        return self.df[self.name].to_numpy()

    @property
    def unique_values(self) -> ndarray:
        self._assert_not_empty()
        return self.df[self.name].unique()

    @property
    def value_counts(self) -> Dict:
        self._assert_not_empty()
        return self.df[self.name].value_counts().to_dict()


@dataclass
class SEColumn(Column):
    """
    Standard deviation column
    """

    def _check_df_values(self, df: DataFrame):
        super()._check_df_values(df)
        if any(df[self.name] <= 0):
            raise ValueError(f"Column({self.name}) values has to be positive.")


@dataclass
class KeyColumn(Column):
    """
    Key column
    """

    name: str = "key"

    def _set_default_values(self, df: DataFrame):
        df[self.name] = np.arange(df.shape[0])

    def _check_df_values(self, df: DataFrame):
        super()._check_df_values(df)
        if len(df[self.name].unique()) != df.shape[0]:
            raise ValueError(f"Column({self.name}) values has to be unique.")


@dataclass
class GroupColumn(Column):
    """
    Group column
    """

    name: str = "group"

    def _set_default_values(self, df: DataFrame):
        df[self.name] = "unknown"


@dataclass
class InterceptColumn(Column):
    """
    Intercept column
    """

    name: str = "intercept"

    def _set_default_values(self, df: DataFrame):
        df[self.name] = 1.0

    def _check_df_values(self, df: DataFrame):
        super()._check_df_values(df)
        if not np.allclose(df[self.name], 1):
            raise ValueError(f"Column({self.name}) values has to be 1.")


# pylint: disable=too-many-instance-attributes
class MRData:
    """Data for simple linear mixed effects model.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 obs: str = Column.name,
                 obs_se: str = SEColumn.name,
                 group: str = GroupColumn.name,
                 key: str = KeyColumn.name,
                 other_cols: List[str] = None):
        self.obs = Column(obs)
        self.obs_se = SEColumn(obs_se)
        self.group = GroupColumn(group)
        self.key = KeyColumn(key)
        self._df = None

        self.cols = []
        self.col_names = []

        cols = [self.obs, self.obs_se, self.group, self.key]
        if other_cols is not None:
            cols.extend([Column(col_name) for col_name in other_cols])
        cols.append(InterceptColumn)

        for col in cols:
            self._add_column(col)

    def _add_column(self, col: Column):
        if (col.name is not None) and (col.name not in self.col_names):
            self.cols.append(col)
            self.col_names.append(col.name)

    def _get_required_cols(self) -> Dict:
        return dict(
            obs=self.obs.name,
            obs_se=self.obs_se.name,
            group=self.group.name,
            key=self.key.name
        )

    def _get_init_inputs(self) -> Dict:
        required_cols = self._get_required_cols()
        other_cols = [col_name for col_name in self.col_names
                      if col_name not in required_cols.values()]
        return dict(**required_cols, other_cols=other_cols)

    @property
    def df(self) -> DataFrame:
        return self._df

    @df.setter
    def df(self, df: DataFrame):
        for col in self.cols:
            col.df = df
        self._df = df

    @property
    def is_empty(self) -> bool:
        return self.df is None

    def _assert_not_empty(self):
        if self.is_empty:
            raise ValueError("Data object is empty.")

    @property
    def shape(self):
        self._assert_not_empty()
        return self.df.shape

    def sort_values(self, *args, **kwargs):
        self._assert_not_empty()
        kwargs = dict(**kwargs, inplace=True)
        self.df.sort_values(*args, **kwargs)

    @classmethod
    def load(cls, df: DataFrame, **kwargs) -> "MRData":
        instance = cls(**kwargs)
        instance.df = df
        return instance

    def copy(self) -> "MRData":
        return self.__copy__()

    def __getitem__(self, col_names: Union[str, List[str]]) -> ndarray:
        self._assert_not_empty()
        return self.df[col_names].to_numpy()

    def __copy__(self) -> "MRData":
        return type(self)(**self._get_init_inputs())

    def __repr__(self) -> str:
        if self.is_empty:
            summary = "MRData(empty)"
        else:
            summary = f"MRData(shape={self.shape})"
        return summary
