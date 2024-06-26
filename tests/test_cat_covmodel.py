import numpy as np
import pandas as pd
import pytest

from mrtool.core.cov_model import CatCovModel
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
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat")
    assert covmodel.alt_cov == ["alt_cat"]
    assert covmodel.ref_cov == ["ref_cat"]

    covmodel = CatCovModel(alt_cov="alt_cat")
    assert covmodel.alt_cov == ["alt_cat"]
    assert covmodel.ref_cov == []

    with pytest.raises(ValueError):
        CatCovModel(alt_cov=["a", "b"])

    with pytest.raises(ValueError):
        CatCovModel(alt_cov="a", ref_cov=["a", "b"])


def test_attach_data(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat")
    assert not hasattr(covmodel, "cats")
    covmodel.attach_data(data)
    assert covmodel.cats.to_list() == ["A", "B", "C", "D"]


def test_has_data(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat")
    assert not covmodel.has_data()

    covmodel.attach_data(data)
    assert covmodel.has_data()


def test_encode(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat")
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


def test_create_design_mat(data):
    covmodel = CatCovModel(alt_cov="alt_cat", ref_cov="ref_cat")
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
