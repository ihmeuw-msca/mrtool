"""
Ensemble Model
"""
from typing import List, Dict
from copy import deepcopy
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from regmod.utils import SplineSpecs
from mrtool.core.data import MRData
from mrtool.core.cov_model import CovModel
from mrtool.core.model import MRBRT
from mrtool.core.utils import proj_simplex


class MRBeRT:
    """Ensemble model of MRBRT.
    """

    def __init__(self, sub_models: List[MRBRT]):
        if not all([isinstance(model, MRBRT) for model in sub_models]):
            raise TypeError("Must provide a list of MRBRT instances.")

        self.sub_models = list(sub_models)
        self.sub_model_weights = np.full(self.num_sub_models,
                                         1/self.num_sub_models)

        if not all([model.data is self.sub_models[0].data
                    for model in self.sub_models]):
            raise ValueError("All sub models must share the same data object.")

        self.data = self.sub_models[0].data
        self.df = self.data.df

    @property
    def num_sub_models(self) -> int:
        return len(self.sub_models)

    def fit_model(self, **fit_options):
        scores = []
        for model in self.sub_models:
            model.fit_model(**fit_options)
            # for now use the insample objective as the score
            # discuss cross-validation
            scores.append(-model.lt.objective(model.lt.soln))

        self.sub_model_weights = proj_simplex(np.array(scores))

        # gathering information from sub models
        w = np.vstack([model.lt.w for model in self.sub_models])
        self.df.is_outlier = (w.T.dot(self.sub_model_weights) <= 0.1).astype(int)

    def get_prediction(self, df: DataFrame = None, **kwargs) -> ndarray:
        predictions = np.vstack([
            model.get_prediction(df, **kwargs)
            for model in self.sub_models
        ])

        return predictions.T.dot(self.sub_model_weights)

    def get_draws(self,
                  df: DataFrame = None,
                  size: int = 1,
                  **kwargs) -> ndarray:
        sizes = np.random.multinomial(size, self.sub_model_weights)
        return np.vstack([
            model.get_draws(df, size=sizes[i], **kwargs)
            for i, model in enumerate(self.sub_models)
        ])

    @classmethod
    def get_knots_ensemble_model(cls,
                                 data: MRData,
                                 ensemble_fe_cov_model: CovModel,
                                 knots_samples: ndarray,
                                 spline_options: Dict = None,
                                 fe_cov_models: List[CovModel] = None,
                                 re_cov_models: List[CovModel] = None,
                                 inlier_pct: float = 1.0) -> "MRBeRT":
        if spline_options is None:
            spline_options = {}
        if fe_cov_models is None:
            fe_cov_models = []

        spline_specs = [
            SplineSpecs(knots, **spline_options)
            for knots in knots_samples
        ]

        sub_models = []
        for i in range(len(spline_specs)):
            cov_model = deepcopy(ensemble_fe_cov_model)
            cov_model.spline = spline_specs[i]
            sub_fe_cov_models = [cov_model] + deepcopy(fe_cov_models)
            sub_models.append(
                MRBRT(data, sub_fe_cov_models, re_cov_models, inlier_pct)
            )

        return cls(sub_models)
