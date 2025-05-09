"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_above_threshold


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize(
    'test_patient, test_result',
    [
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
    ]
)
def test_daily_above_threshold(test_patient, test_result):
    """Test that threshold function works for an array of positive integers."""
    test_data = np.array([[0, 3, 6, 1],
                          [0, 5, 10, 4],
                          [4, 3, 2, 1],
                          [12, 12, 12, 12]])
    test_threshold = 4

    assert daily_above_threshold(
        test_data, test_patient, test_threshold) == test_result
    # npt.assert_array_equal(
    #     daily_above_threshold(test_data, test_patient, test_threshold),
    #     test_result
    # )
