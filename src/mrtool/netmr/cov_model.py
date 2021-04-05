"""
Network Covariate Model
"""
from typing import Callable
import numpy as np
from scipy.linalg import block_diag
from mrtool.netmr.data import NetMRData
from mrtool.core.cov_model import LinearCovModel
from mrtool.core.utils import mat_to_fun


class NetLinearCovModel(LinearCovModel):
    """
    Net Linear Covariates Model
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_dorms = 1

    @property
    def size(self) -> int:
        return super().size*self.num_dorms

    def attach_data(self, data: NetMRData):
        super().attach_data(data)
        self.num_dorms = len(data.unique_dorms)

    def get_prior_array(self, ptype: str):
        mat, vec = super().get_prior_array(ptype)
        if mat is None:
            if vec.shape[1] == super().size:
                vec = np.tile(vec, (1, self.num_dorms))
        else:
            if mat.shape[1] == super().size:
                mat = block_diag(*[mat]*self.num_dorms)
                vec = np.tile(vec, (1, self.num_dorms))
        return mat, vec

    def get_fun(self, data: NetMRData) -> Callable:
        relation_mat = data.get_relation_mat()
        cov_mat = self.get_mat(data)

        mat = (
            relation_mat.ravel()[:, None] *
            np.repeat(cov_mat, relation_mat.shape[1], axis=0)
        ).reshape(data.shape[0], relation_mat.shape[1]*cov_mat.shape[1])

        return mat_to_fun(mat)
