# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    Model module for mrtool package.
"""
import operator
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
from pandas import DataFrame
from mrtool.core import utils
from mrtool.core.cov_model import CovModel
from mrtool.core.data import MRData
from mrtool.core.limetr_api import get_limetr, get_soln
from numpy import ndarray
from scipy.linalg import block_diag


# pylint:disable=too-many-instance-attributes
class MRBRT:

    def __init__(self,
                 data: MRData,
                 fe_cov_models: Union[CovModel, Iterable[CovModel]],
                 re_cov_models: Union[CovModel, Iterable[CovModel]] = None,
                 inlier_pct: float = 1.0):
        self.data = data
        self.fe_cov_models = fe_cov_models
        self.re_cov_models = re_cov_models
        self.inlier_pct = inlier_pct

        self.df = self.data.df
        self.attach_data()

        self.lt = get_limetr(self)
        self.soln = None

    data = property(operator.attrgetter("_data"))

    @data.setter
    def data(self, data: MRData):
        if not isinstance(data, MRData):
            raise TypeError("data has to be instance of MRData.")
        if data.is_empty:
            raise ValueError("data must not be empty.")
        self._data = data

    fe_cov_models = property(operator.attrgetter("_fe_cov_models"))

    # pylint:disable=isinstance-second-argument-not-valid-type
    @fe_cov_models.setter
    def fe_cov_models(self, models: Union[CovModel, Iterable[CovModel]]):
        if isinstance(models, CovModel):
            self._fe_cov_models = [models]
        elif isinstance(models, Iterable):
            if not all([isinstance(model, CovModel) for model in models]):
                raise TypeError("All models in list must be CovModel.")
            self._fe_cov_models = list(models)
        else:
            raise TypeError("models must be one or a list of CovModel.")

    re_cov_models = property(operator.attrgetter("_re_cov_models"))

    @re_cov_models.setter
    def re_cov_models(self, models: Union[CovModel, Iterable[CovModel]]):
        if isinstance(models, CovModel):
            self._re_cov_models = [models]
        elif isinstance(models, Iterable):
            if not all([isinstance(model, CovModel) for model in models]):
                raise TypeError("All models in list must be CovModel.")
            self._re_cov_models = list(models)
        elif models is None:
            self._re_cov_models = []
        else:
            raise TypeError("models must be one or a list of CovModel or None.")

    inlier_pct = property(operator.attrgetter("_inlier_pct"))

    @inlier_pct.setter
    def inlier_pct(self, pct: float):
        if pct < 0.0 or pct > 1.0:
            raise ValueError("inlier_pct must be between 0 and 1.")
        self._inlier_pct = pct

    def attach_data(self, data: MRData = None):
        """Attach data to cov_model.
        """
        data = self.data if data is None else data
        # attach data to cov_model
        for cov_model in self.fe_cov_models + self.re_cov_models:
            cov_model.attach_data(data)

    @property
    def fe_sizes(self) -> List[int]:
        return [model.size for model in self.fe_cov_models]

    @property
    def fe_size(self) -> int:
        return sum(self.fe_sizes)

    @property
    def re_sizes(self) -> List[int]:
        return [model.size for model in self.re_cov_models]

    @property
    def re_size(self) -> int:
        return sum(self.re_sizes)

    @property
    def size(self) -> int:
        return self.fe_size + self.re_size

    @property
    def fe_fun(self) -> Tuple[Callable, Callable]:
        funs = [model.get_fun(self.data) for model in self.fe_cov_models]
        funs, jac_funs = tuple(zip(*funs))
        indicies = utils.sizes_to_slices(self.fe_sizes)

        def fun(beta: ndarray) -> ndarray:
            return sum([funs[i](beta[index])
                        for i, index in enumerate(indicies)])

        def jac_fun(beta: ndarray) -> ndarray:
            return np.hstack([jac_funs[i](beta[index])
                              for i, index in enumerate(indicies)])
        return fun, jac_fun

    @property
    def re_mat(self) -> ndarray:
        if len(self.re_cov_models) == 0:
            return np.empty(shape=(self.data.shape[0], 0))
        return np.hstack([model.get_mat() for model in self.re_cov_models])

    def get_priors(self, ptype: str):
        priors = [model.get_priors(ptype)
                  for model in self.fe_cov_models + self.re_cov_models]
        if "linear" not in ptype:
            result = np.hstack(priors)
        else:
            mat = block_diag(*[prior[0] for prior in priors])
            vec = np.hstack([prior[1] for prior in priors])
            result = (mat, vec)
        return result

    def fit_model(self, **fit_options):
        self.lt.fitModel(**fit_options)
        self.df["is_outlier"] = (self.lt.w <= 0.1).astype(int)
        self.soln = get_soln(self)

    def get_prediction(self, df: DataFrame = None, **kwargs) -> ndarray:
        df = self.df if df is None else df
        self.data.df = df
        fe_fun, _ = self.fe_fun
        re_mat = self.re_mat
        pred = self.soln.predict(fe_fun, re_mat, **kwargs)
        self.data.df = self.df
        return pred

    def get_draws(self, df: DataFrame = None, **kwargs) -> ndarray:
        df = self.df if df is None else df
        self.data.df = df
        fe_fun, _ = self.fe_fun
        re_mat = self.re_mat
        draws = self.soln.get_draws(fe_fun, re_mat, **kwargs)
        self.data.df = self.df
        return draws
