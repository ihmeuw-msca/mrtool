# -*- coding: utf-8 -*-
"""
    data
    ~~~~

    `data` module of the `crosswalk` package.
"""
from typing import Dict
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from mrtool.core.data import Column, MRData


@dataclass
class DormColumn(Column):
    """
    Dorm Column
    """

    separator: str = None

    def _set_default_values(self, df: DataFrame):
        df[self.name] = self.name

    @property
    def values(self) -> ndarray:
        self._assert_not_empty()
        return self.df[self.name].str.split(pat=self.separator).to_numpy()

    @property
    def unique_values(self) -> ndarray:
        return np.unique(np.hstack(self.values))

    @property
    def value_counts(self) -> Dict:
        values, counts = np.unique(np.hstack(self.values), return_counts=True)
        return dict(zip(values, counts))


@dataclass
class RefDormColumn(DormColumn):
    """
    Reference Dorm Column
    """

    name: str = "ref_dorm"


@dataclass
class AltDormColumn(DormColumn):
    """
    Alternative Dorm Column
    """

    name: str = "alt_dorm"


class NetMRData(MRData):
    """Data for network meta analysis model.
    """

    def __init__(self,
                 ref_dorm: str = RefDormColumn.name,
                 alt_dorm: str = AltDormColumn.name,
                 **kwargs):
        super().__init__(**kwargs)
        self.ref_dorm = RefDormColumn(ref_dorm)
        self.alt_dorm = AltDormColumn(alt_dorm)
        self._add_column(self.ref_dorm)
        self._add_column(self.alt_dorm)
