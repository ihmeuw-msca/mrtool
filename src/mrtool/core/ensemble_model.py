"""
Ensemble Model
"""
from typing import List
import numpy as np
from numpy import ndarray
from pandas import DataFrame
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
            scores.append(-model.lt.objective(model.lt.soln))

        self.sub_model_weights = proj_simplex(np.array(scores))

        # gathering information from sub models
        w = np.vstack([model.lt.w for model in self.sub_models])
        self.df.is_outlier = (w.T.dot(self.sub_model_weights) <= 0.1).astype(int)

    def predict(self, df: DataFrame = None, **kwargs) -> ndarray:
        predictions = [
            model.predict(df, **kwargs)
            for model in self.sub_models
        ]

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
