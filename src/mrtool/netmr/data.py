# -*- coding: utf-8 -*-
"""
    data
    ~~~~

    `data` module of the `crosswalk` package.
"""
from typing import Dict
from collections import Counter
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
    # need memory of the dorm structure

    def __init__(self,
                 ref_dorm: str = RefDormColumn.name,
                 alt_dorm: str = AltDormColumn.name,
                 dorm_separator: str = DormColumn.separator,
                 **kwargs):
        super().__init__(**kwargs)
        self.ref_dorm = RefDormColumn(ref_dorm, separator=dorm_separator)
        self.alt_dorm = AltDormColumn(alt_dorm, separator=dorm_separator)
        self.add_column(self.ref_dorm)
        self.add_column(self.alt_dorm)

    @property
    def unique_dorms(self) -> ndarray:
        self._assert_not_empty()
        return np.unique(np.hstack([self.ref_dorm.unique_values,
                                    self.alt_dorm.unique_values]))

    @property
    def dorm_counts(self) -> Dict:
        counter = Counter()
        counter.update(self.ref_dorm.value_counts)
        counter.update(self.alt_dorm.value_counts)
        return dict(counter)

    def get_relation_mat(self) -> ndarray:
        self._assert_not_empty()
        dorms = list(self.unique_dorms)
        mat = np.zeros(self.shape[0], len(dorms))

        for i in range(self.shape[0]):
            for dorm in self.alt_dorm.values:
                mat[i, dorms.index(dorm)] += 1.0
            for dorm in self.ref_dorm.values:
                mat[i, dorms.index(dorm)] -= 1.0

        return mat
