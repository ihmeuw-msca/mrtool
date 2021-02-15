# -*- coding: utf-8 -*-
"""
    data
    ~~~~

    `data` module of the `crosswalk` package.
"""
from typing import List
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import warnings
from mrtool.xwalk.utils import array_structure, process_dorms
from mrtool.core.data import MRData
from mrtool.core.utils import empty_array, expand_array


@dataclass
class XData(MRData):
    alt_dorms: List[List[str]] = field(default_factory=empty_array)
    ref_dorms: List[List[str]] = field(default_factory=empty_array)

    def __post_init__(self):
        super().__post_init__()
        if len(self.alt_dorms) == 0:
            self.alt_dorms = [["alt"]]*self.num_points
        if len(self.ref_dorms) == 0:
            self.ref_dorms = [["ref"]]*self.num_points

        self._get_dorm_structure()

    def _get_dorm_structure(self):
        (self.num_dorms,
         self.dorm_sizes,
         self.unique_dorms) = array_structure(np.hstack([self.alt_dorms, self.ref_dorms]))
        (self.num_alt_dorms,
         self.alt_dorm_sizes,
         self.unique_alt_dorms) = array_structure(self.alt_dorms)
        (self.num_ref_dorms,
         self.ref_dorm_sizes,
         self.unique_ref_dorms) = array_structure(self.ref_dorms)

        self.dorm_idx = {dorm: i
                         for i, dorm in enumerate(self.unique_dorms)}

    @property
    def max_dorm(self) -> str:
        return self.unique_dorms[np.argmax(self.dorm_sizes)]

    @property
    def min_dorm(self) -> str:
        return self.unique_dorms[np.argmin(self.dorm_sizes)]

    @property
    def max_alt_dorm(self) -> str:
        return self.unique_alt_dorms[np.argmax(self.alt_dorm_sizes)]

    @property
    def min_ref_dorm(self) -> str:
        return self.unique_alt_dorms[np.argmin(self.alt_dorm_sizes)]

    @property
    def max_ref_dorm(self) -> str:
        return self.unique_ref_dorms[np.argmax(self.ref_dorm_sizes)]

    @property
    def min_ref_dorm(self) -> str:
        return self.unique_ref_dorms[np.argmin(self.ref_dorm_sizes)]

    def reset(self):
        super().reset()
        self.alt_dorms = empty_array()
        self.ref_dorms = empty_array()

    def load_df(self,
                data: pd.DataFrame,
                col_obs: str = None,
                col_obs_se: str = None,
                col_covs: List[str] = None,
                col_study_id: str = None,
                col_data_id: str = None,
                col_alt_dorms: str = None,
                col_ref_dorms: str = None,
                dorm_separator: str = None):
        super().load_df(data,
                        col_obs,
                        col_obs_se,
                        col_covs,
                        col_study_id,
                        col_data_id)

        alt_dorms = None if col_alt_dorms is None else data[col_alt_dorms].to_numpy().astype(str)
        ref_dorms = None if col_ref_dorms is None else data[col_ref_dorms].to_numpy().astype(str)

        self.alt_dorms = process_dorms(dorms=alt_dorms, size=self.num_points,
                                       default_dorm="alt", dorm_separator=dorm_separator)
        self.ref_dorms = process_dorms(dorms=ref_dorms, size=self.num_points,
                                       default_dorm="ref", dorm_separator=dorm_separator)
        self._get_dorm_structure()

    def __repr__(self):
        return (f"number of observations: {self.num_obs}\n"
                f"number of covariates  : {self.num_covs}\n"
                f"number of defs/methods: {self.num_dorms}\n"
                f"number of studies     : {self.num_studies}")
