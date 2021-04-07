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


@pytest.mark.parametrize(("x", "result"),
                         [(np.ones(3), np.ones(3)/3),
                          (np.array([0.0, 1.0]), np.array([0.0, 1.0]))])
def test_proj_simplex(x, result):
    my_result = utils.proj_simplex(x)
    assert np.allclose(my_result, result)


@pytest.mark.parametrize("num_knots", [2, 3, 4])
@pytest.mark.parametrize("sample_width", [0.1, 0.2])
@pytest.mark.parametrize("sample_size", [1, 5])
def test_sample_knots(num_knots, sample_width, sample_size):
    knots_samples = utils.sample_knots(num_knots, sample_width, sample_size)
    assert knots_samples.shape == (sample_size, num_knots)
    intervals = np.diff(knots_samples, axis=1)
    assert np.all(intervals >= 0)
