from unittest.mock import Mock
import numpy as np


def test_analyse_data_mock_source():
    from inflammation.compute_data import analyse_data
    data_source = Mock()
    mock_data = [
        np.array([[1, 2, 4],
                  [3, 4, 8]]),
        np.array([[5, 6, 12],
                  [6, 7, 14]])
    ]
    data_source.load_inflammation_data.return_value = mock_data

    analyse_data(data_source)
