# -*- coding: utf-8 -*-
"""
    test_utils
"""
import numpy as np
import pytest
from mrtool.core import utils


@pytest.mark.parametrize('sizes', [np.array([1, 2, 3])])
@pytest.mark.parametrize('indices', [[np.arange(0, 1),
                                      np.arange(1, 3),
                                      np.arange(3, 6)]])
def test_sizes_to_indices(sizes, indices):
    my_indices = utils.sizes_to_indices(sizes)
    assert all([np.allclose(my_indices[i], indices[i])
                for i in range(len(sizes))])


@pytest.mark.parametrize('obj', [1, 1.0, 'a', True, [1], [1.0], ['a'], [True]])
def test_to_list(obj):
    obj_list = utils.to_list(obj)
    if isinstance(obj, list):
        assert obj_list is obj
    else:
        assert isinstance(obj_list, list)


@pytest.mark.parametrize(("x", "result"),
                         [(np.ones(3), np.ones(3)/3),
                          (np.array([0.0, 1.0]), np.array([0.0, 1.0]))])
def test_proj_simplex(x, result):
    my_result = utils.proj_simplex(x)
    assert np.allclose(my_result, result)
