import numpy as np
import pandas as pd
import pytest

from mrtool.core.cov_model import CatCovModel, LinearCatCovModel, LogCatCovModel
from mrtool.core.data import MRData


@pytest.fixture
def data():
    df = pd.DataFrame(
        dict(
            obs=[0, 1, 0, 1],
            obs_se=[0.1, 0.1, 0.1, 0.1],
            alt_cat=["A", "A", "B", "C"],
            ref_cat=["A", "B", "B", "D"],
            study_id=[1, 1, 2, 2],
        )
    )
    data = MRData()
    data.load_df(
        df,
        col_obs="obs",
        col_obs_se="obs_se",
        col_covs=["alt_cat", "ref_cat"],
        col_study_id="study_id",
    )
    return data


def test_init():
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    assert covmodel.alt_cov == ["alt_cat"]
    assert covmodel.ref_cov == ["ref_cat"]

    covmodel = CatCovModel(alt_cov="alt_cat")
    assert covmodel.alt_cov == ["alt_cat"]
    assert covmodel.ref_cov == []

    with pytest.raises(ValueError):
        CatCovModel(alt_cov=["a", "b"])

    with pytest.raises(ValueError):
        CatCovModel(alt_cov="a", ref_cov=["a", "b"], ref_cat="A")


def test_attach_data(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    assert not hasattr(covmodel, "cats")
    covmodel.attach_data(data)
    assert covmodel.cats.to_list() == ["A", "B", "C", "D"]


def test_ref_cov(data):
    with pytest.raises(ValueError):
        covmodel = CatCovModel(
            alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="E"
        )
        covmodel.attach_data(data)

    with pytest.raises(ValueError):
        covmodel = CatCovModel(alt_cov="alt_cat", ref_cat="A")

    covmodel = CatCovModel(alt_cov="alt_cat")
    covmodel.attach_data(data)
    assert covmodel.ref_cat is None
    assert np.isinf(covmodel.prior_beta_uniform).all()

    covmodel = CatCovModel(
        alt_cov="alt_cat", use_re=True, use_re_intercept=False
    )
    covmodel.attach_data(data)
    assert covmodel.ref_cat is None
    assert np.isinf(covmodel.prior_beta_uniform).all()
    assert np.allclose(covmodel.prior_gamma_uniform[0], 0.0)
    assert np.isposinf(covmodel.prior_gamma_uniform[1]).all()

    with pytest.warns():
        covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat")
    assert covmodel.ref_cat is None
    covmodel.attach_data(data)
    assert covmodel.ref_cat == "A"

    covmodel = CatCovModel(
        alt_cov="alt_cat",
        ref_cov="ref_cat",
        ref_cat="B",
        use_re=True,
        use_re_intercept=False,
    )
    covmodel.attach_data(data)
    assert covmodel.ref_cat == "B"
    my_beta_uprior = np.array(
        [[-np.inf, 0.0, -np.inf, -np.inf], [np.inf, 0.0, np.inf, np.inf]]
    )
    my_gamma_uprior = np.array(
        [[0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, np.inf, np.inf]]
    )
    assert np.allclose(covmodel.prior_beta_uniform, my_beta_uprior)
    assert np.allclose(covmodel.prior_gamma_uniform, my_gamma_uprior)


def test_has_data(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    assert not covmodel.has_data()

    covmodel.attach_data(data)
    assert covmodel.has_data()


def test_encode(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    covmodel.attach_data(data)

    mat = covmodel.encode(["A", "B", "C", "C"])
    true_mat = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ]
    )
    assert np.allclose(mat, true_mat)


def test_encode_fail(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    covmodel.attach_data(data)

    with pytest.raises(ValueError):
        covmodel.encode(["A", "B", "C", "E"])


def test_create_design_mat(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    covmodel.attach_data(data)

    alt_mat, ref_mat = covmodel.create_design_mat(data)

    assert np.allclose(
        alt_mat,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    )

    assert np.allclose(
        ref_mat,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_order_prior(data):
    covmodel = CatCovModel(
        alt_cov="alt_cat",
        ref_cov="ref_cat",
        ref_cat="A",
        prior_order=[["A", "B"], ["B", "C"]],
    )
    covmodel.attach_data(data)
    assert covmodel.prior_order == [("A", "B"), ("B", "C")]

    covmodel = CatCovModel(
        alt_cov="alt_cat",
        ref_cov="ref_cat",
        ref_cat="A",
        prior_order=[["A", "B", "C"], ["B", "C"]],
    )
    assert covmodel.prior_order == [("A", "B"), ("B", "C")]

    with pytest.raises(ValueError):
        covmodel = CatCovModel(
            alt_cov="alt_cat",
            ref_cov="ref_cat",
            ref_cat="A",
            prior_order=[["A", "B"], ["B", "C", "E"]],
        )
        covmodel.attach_data(data)


def test_num_x_vars(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    assert covmodel.num_x_vars == 0
    covmodel.attach_data(data)
    assert covmodel.num_x_vars == 4


def test_num_z_vars(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    assert covmodel.num_z_vars == 0

    covmodel = CatCovModel(
        alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A", use_re=True
    )
    assert covmodel.num_z_vars == 1

    covmodel = CatCovModel(
        alt_cov="alt_cat",
        ref_cov="ref_cat",
        ref_cat="A",
        use_re=True,
        use_re_intercept=False,
    )
    assert covmodel.num_z_vars == 0
    covmodel.attach_data(data)
    assert covmodel.num_z_vars == 4


def test_num_constraints(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    assert covmodel.num_constraints == 0
    covmodel = CatCovModel(
        alt_cov="alt_cat",
        ref_cov="ref_cat",
        ref_cat="A",
        prior_order=[["A", "B", "C"]],
    )
    assert covmodel.num_constraints == 2


def test_create_constraint_mat(data):
    covmodel = CatCovModel(
        alt_cov="alt_cat",
        ref_cov="ref_cat",
        ref_cat="A",
        prior_order=[["A", "B", "C"]],
    )
    covmodel.attach_data(data)
    c_mat, c_val = covmodel.create_constraint_mat()
    assert np.allclose(
        c_mat,
        np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
            ]
        ),
    )

    assert np.allclose(
        c_val,
        np.array(
            [
                [-np.inf, -np.inf],
                [0.0, 0.0],
            ]
        ),
    )


def test_z_mat(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    covmodel.attach_data(data)
    z_mat = covmodel.create_z_mat(data)
    assert z_mat.shape == (data.num_obs, 0)

    covmodel = CatCovModel(
        alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A", use_re=True
    )
    covmodel.attach_data(data)
    z_mat = covmodel.create_z_mat(data)
    assert np.allclose(z_mat, np.ones((data.num_obs, 1)))

    covmodel = CatCovModel(
        alt_cov="alt_cat",
        ref_cov="ref_cat",
        ref_cat="A",
        use_re=True,
        use_re_intercept=False,
    )
    covmodel.attach_data(data)
    z_mat = covmodel.create_z_mat(data)
    assert np.allclose(
        z_mat,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )


def test_linearcatcovmodel_create_x_fun(data):
    np.random.seed(0)

    x = np.random.randn(3)
    covmodel = LinearCatCovModel(alt_cov="alt_cat")
    covmodel.attach_data(data)
    x_fun, jac_fun = covmodel.create_x_fun(data)
    x_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(x_fun(x), x_mat @ x)
    assert np.allclose(jac_fun(x), x_mat)

    x = np.random.randn(4)
    covmodel = LinearCatCovModel(
        alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A"
    )
    covmodel.attach_data(data)
    x_fun, jac_fun = covmodel.create_x_fun(data)
    x_mat = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )
    assert np.allclose(x_fun(x), x_mat @ x)
    assert np.allclose(jac_fun(x), x_mat)


def test_logcatcovmodel_attach_data(data):
    covmodel = LogCatCovModel(alt_cov="alt_cat")
    covmodel.attach_data(data)
    assert np.allclose(
        covmodel.prior_beta_uniform,
        np.repeat(np.array([[1e-6], [np.inf]]), 3, axis=1),
    )

    covmodel = LogCatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    covmodel.attach_data(data)
    assert np.allclose(
        covmodel.prior_beta_uniform,
        np.array([[0.0] + [-1 + 1e-6] * 3, [0.0] + [np.inf] * 3]),
    )


def test_logcatcovmodel_create_x_fun(data):
    np.random.seed(0)

    x = np.random.rand(3)
    covmodel = LogCatCovModel(alt_cov="alt_cat")
    covmodel.attach_data(data)
    x_fun, _ = covmodel.create_x_fun(data)
    x_vec = np.log(x[[0, 0, 1, 2]])
    assert np.allclose(x_fun(x), x_vec)

    x = np.random.rand(4)
    covmodel = LogCatCovModel(alt_cov="alt_cat", ref_cov="ref_cat", ref_cat="A")
    covmodel.attach_data(data)
    x_fun, _ = covmodel.create_x_fun(data)
    x_vec = np.log(1.0 + x[[0, 0, 1, 2]]) - np.log(1 + x[[0, 1, 1, 3]])
    assert np.allclose(x_fun(x), x_vec)
