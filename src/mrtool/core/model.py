# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    Model module for mrtool package.
"""
import operator
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
from mrtool.core import utils
from mrtool.core.cov_model import CovModel
from mrtool.core.data import MRData
from mrtool.core.limetr_api import get_limetr
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
        indicies = utils.sizes_to_sclices(self.fe_sizes)

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
            return np.empty(shape=(self.df.shape[0], 0))
        return np.hstack([model.get_mat() for model in self.re_cov_models])

    @property
    def uvec(self) -> ndarray:
        return np.hstack([
            model.get_uvec()
            for model in self.fe_cov_models + self.re_cov_models
        ])

    @property
    def gvec(self) -> ndarray:
        return np.hstack([
            model.get_gvec()
            for model in self.fe_cov_models + self.re_cov_models
        ])

    @property
    def linear_umat(self) -> ndarray:
        return block_diag(*[
            model.get_linear_umat()
            for model in self.fe_cov_models + self.re_cov_models
        ])

    @property
    def linear_uvec(self) -> ndarray:
        return np.hstack([
            model.get_linear_uvec()
            for model in self.fe_cov_models + self.re_cov_models
        ])

    @property
    def linear_gmat(self) -> ndarray:
        return block_diag(*[
            model.get_linear_gmat()
            for model in self.fe_cov_models + self.re_cov_models
        ])

    @property
    def linear_gvec(self) -> ndarray:
        return np.hstack([
            model.get_linear_gvec()
            for model in self.fe_cov_models + self.re_cov_models
        ])

    def fit_model(self, **fit_options):
        self.lt.fitModel(**fit_options)
        self.df["is_outlier"] = (self.lt.w <= 0.1).astype(int)

    def predict(self) -> np.ndarray:
        pass

    def get_draws(self) -> np.ndarray:
        pass
