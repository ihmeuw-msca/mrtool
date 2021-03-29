"""
Network Covariate Model
"""
from typing import Callable
import numpy as np
from mrtool.netmr.data import NetMRData
from mrtool.core.cov_model import LinearCovModel
from mrtool.core.utils import mat_to_fun


class NetLinearCovModel(LinearCovModel):
    """
    Net Linear Covariates Model
    """
    # size
    # priors

    def get_fun(self, data: NetMRData) -> Callable:
        relation_mat = data.get_relation_mat()
        cov_mat = self.get_mat(data)

        mat = (
            relation_mat.ravel()[:, None] *
            np.repeat(cov_mat, len(relation_mat.shape[1]))
        ).reshape(data.shape[0], relation_mat.shape[1]*cov_mat.shape[1])

        return mat_to_fun(mat)
