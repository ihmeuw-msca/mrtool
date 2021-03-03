"""
Interface for LimeTr
"""
from pkg_resources import working_set
import numpy as np
from mrtool.core.utils import mat_to_fun

# pylint:disable=not-an-iterable
installed_pkgs = [p.key for p in working_set]
# pylint:disable=import-error
if "limetr" in installed_pkgs:
    from limetr import LimeTr
else:
    # create fake LimeTr class
    class LimeTr:
        def __init__(self, n, k_beta, k_gamma, Y, F, JF, Z,
                     S=None, share_obs_std=False,
                     C=None, JC=None, c=None,
                     H=None, JH=None, h=None,
                     uprior=None, gprior=None, lprior=None,
                     certain_inlier_id=None,
                     inlier_percentage=1.0):
            pass


def get_limetr(model: "MRBRT") -> LimeTr:
    # dimensions
    data = model.data
    data.sort_values(data.group.name)
    group_unique_values = data.group.unique_values
    group_value_counts = data.group.value_counts
    n = [group_value_counts[v] for v in group_unique_values]
    k_beta = model.fe_size
    k_gamma = model.re_size

    # create x fun and z mat
    x_fun, x_fun_jac = model.fe_fun
    z_mat = model.re_mat

    # priors
    uvec = model.uvec
    gvec = model.gvec
    linear_ufun, linear_ujac_fun = mat_to_fun(model.linear_umat)
    linear_uvec = model.linear_uvec
    linear_gfun, linear_gjac_fun = mat_to_fun(model.linear_gmat)
    linear_gvec = model.linear_gvec

    # consider the situation when there is not random effects
    if k_gamma == 0:
        k_gamma = 1
        z_mat = np.ones((data.shape[0], 1))
        uvec = np.hstack([uvec, np.array([[0.0], [0.0]])])
        gvec = np.hstack([gvec, np.array([[0.0], [np.inf]])])

    return LimeTr(n,
                  k_beta,
                  k_gamma,
                  data.obs,
                  x_fun,
                  x_fun_jac,
                  z_mat,
                  S=data.obs_se,
                  C=linear_ufun,
                  JC=linear_ujac_fun,
                  c=linear_uvec,
                  H=linear_gfun,
                  JH=linear_gjac_fun,
                  h=linear_gvec,
                  uprior=uvec,
                  gprior=gvec,
                  inlier_percentage=model.inlier_pct)
