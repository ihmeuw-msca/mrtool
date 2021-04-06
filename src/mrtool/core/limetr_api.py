"""
Interface for LimeTr
"""
from pkg_resources import working_set
import numpy as np
from mrtool.core.utils import mat_to_fun
from mrtool.core.soln import MRSoln

# pylint:disable=not-an-iterable
installed_pkgs = [p.key for p in working_set]
# pylint:disable=import-error
if "limetr" in installed_pkgs:
    from limetr import LimeTr
    from limetr.utils import VarMat
else:
    # create fake LimeTr class
    class LimeTr:
        def __init__(self, *args, **kwargs):
            pass

    class VarMat:
        def __init__(self, *args, **kwargs):
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
    x_fun, x_fun_jac = model.get_fe_fun()
    z_mat = model.get_re_mat()

    # priors
    uvec = model.get_priors("uprior")
    gvec = model.get_priors("gprior")
    linear_uprior = model.get_priors("linear_uprior")
    linear_gprior = model.get_priors("linear_gprior")
    linear_ufun, linear_ujac_fun = mat_to_fun(linear_uprior[0])
    linear_uvec = linear_uprior[1]
    linear_gfun, linear_gjac_fun = mat_to_fun(linear_gprior[0])
    linear_gvec = linear_gprior[1]

    # consider the situation when there is no random effects
    if k_gamma == 0:
        k_gamma = 1
        z_mat = np.ones((data.shape[0], 1))
        uvec = np.hstack([uvec, np.array([[0.0], [0.0]])])
        gvec = np.hstack([gvec, np.array([[0.0], [np.inf]])])

    return LimeTr(n,
                  k_beta,
                  k_gamma,
                  data.obs.values,
                  x_fun,
                  x_fun_jac,
                  z_mat,
                  S=data.obs_se.values,
                  C=linear_ufun,
                  JC=linear_ujac_fun,
                  c=linear_uvec,
                  H=linear_gfun,
                  JH=linear_gjac_fun,
                  h=linear_gvec,
                  uprior=uvec,
                  gprior=gvec,
                  inlier_percentage=model.inlier_pct)


def get_soln(model: "MRBRT") -> MRSoln:
    lt = model.lt
    beta = lt.beta
    gamma = lt.gamma

    # compute vcov matrices
    femat = lt.JF(beta)*np.sqrt(model.lt.w)[:, None]
    remat = lt.Z*np.sqrt(model.lt.w)[:, None]
    varmat = VarMat(lt.S**(2*lt.w), remat, gamma, model.lt.n)
    beta_hessian = femat.T.dot(varmat.invDot(femat)) + np.diag(lt.gw[:lt.k_beta])
    gamma_hessian = lt.get_gamma_fisher(gamma)
    beta_vcov = np.linalg.inv(beta_hessian)
    gamma_vcov = np.linalg.inv(gamma_hessian)

    # compute random effects
    u = lt.estimateRE()
    random_effects = {
        g: u[i]
        for i, g in enumerate(model.data.group.unique_values)
    }

    return MRSoln(beta, gamma, beta_vcov, gamma_vcov, random_effects)
