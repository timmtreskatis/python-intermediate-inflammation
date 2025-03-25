"""Tests for the compute_data module
"""
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.compute_data import CSVDataSource, analyse_data, compute_standard_deviation_by_day


def test_analyse_data_mock_source():
    """Tests whether the analyse_data function completes successfully when mock data is passed
    """
    data_source = Mock()
    mock_data = [
        np.array([[1, 2, 4],
                  [3, 4, 8]]),
        np.array([[5, 6, 12],
                  [6, 7, 14]])
    ]
    data_source.load_inflammation_data.return_value = mock_data

    analyse_data(data_source)


def test_analyse_data():
    """Regression test for the analyse_data function
    """
    path = Path.cwd() / "./data"
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)
    expected_result = np.array([0., 0.22510286, 0.18157299, 0.1264423, 0.9495481,
                                0.27118211, 0.25104719, 0.22330897, 0.89680503, 0.21573875,
                                1.24235548, 0.63042094, 1.57511696, 2.18850242, 0.3729574,
                                0.69395538, 2.52365162, 0.3179312, 1.22850657, 1.63149639,
                                2.45861227, 1.55556052, 2.8214853, 0.92117578, 0.76176979,
                                2.18346188, 0.55368435, 1.78441632, 0.26549221, 1.43938417,
                                0.78959769, 0.64913879, 1.16078544, 0.42417995, 0.36019114,
                                0.80801707, 0.50323031, 0.47574665, 0.45197398, 0.22070227])

    npt.assert_array_almost_equal(result, expected_result)


@pytest.mark.parametrize(
    'data, expected_output',
    [
        ([np.array([[0, 1, 2, 0], [0, 2, 3, 1]]),
          np.array([[1, 1, 1, 0], [0, 0, 10, 1]])],
          np.array([0.25, 0.5, 1.5, 0])),
        ([np.array([[1, 2, 3], [0, 10, 0], [1, 2, 0], [1, 4, 1]])],
         np.array([0, 0, 0])),
        ([np.array([[0, 0, 0]]),
          np.array([[0, 10, 0]]),
          np.array([[1, 2, 0]]),
          np.array([[1, 4, 1]])],
          np.array([0.5, np.sqrt(14), np.sqrt(3)/4]))
    ],
    ids=[
        '2 files with 2 patients each',
        '1 file with 4 patients',
        '4 files with 1 patient each'
    ]
)
def test_compute_standard_deviation_by_day(data, expected_output):
    """Tests the computation of standard deviations
    """
    actual_output = compute_standard_deviation_by_day(data)
    npt.assert_array_almost_equal(actual_output, expected_output)
