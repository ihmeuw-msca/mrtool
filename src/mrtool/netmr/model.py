# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    `model` module of the `crosswalk` package.
"""
from mrtool.core.model import MRBRT
from mrtool.netmr.cov_model import NetLinearCovModel


class NetModel(MRBRT):
    """Network Meta-Regression Model
    """

    def __init__(self, *args, gold_dorm: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        if gold_dorm is None:
            k, v = zip(*self.data.dorm_counts.items())
            gold_dorm = k[v.index(max(v))]
        if gold_dorm not in self.data.unique_dorms:
            raise ValueError("Gold dorm doesn't appear in data.")
        self.gold_dorm = gold_dorm

        index = 0
        order = self.data.unique_dorms.index(gold_dorm)
        for cov_model in self.fe_cov_models + self.re_cov_models:
            if isinstance(cov_model, NetLinearCovModel):
                size = super(NetLinearCovModel, cov_model).size
                start = index + order*size
                end = start + size
                self.lt.uprior[:, start:end] = 0.0
            index += cov_model.size
