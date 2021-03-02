# -*- coding: utf-8 -*-
"""
    test_utils
    ~~~~~~~~~~
    Test `utils` module of `sfma` package.
"""
import numpy as np
import pytest
from mrtool import utils


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
