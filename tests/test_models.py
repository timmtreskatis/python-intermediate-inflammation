"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ]
)
def test_daily_mean(test, expected):
    """Test that mean function works for different arrays of integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2, 3, 4], [3, 4, 5, 5], [4, 3, 2, 1] ], [4, 4, 5, 5]),
        ([ [1, 2, 3], [3, 4, 3], [3, 2, 1] ], [3, 4, 3]),
    ]
)
def test_daily_max(test, expected):
    """Test that max function works for different arrays of integers."""
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2, 3, 4], [3, 4, 5, 5], [4, 3, 2, 1] ], [1, 2, 2, 1]),
        ([ [1, 2, 3], [3, 4, 3], [3, 2, 1] ], [1, 2, 1]),
    ]
)
def test_daily_min(test, expected):
    """Test that min function works for different arrays of integers."""
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        daily_min([['Hello', 'there'], ['General', 'Kenobi']])
