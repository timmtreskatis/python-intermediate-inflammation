"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import json
from functools import reduce

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def load_json(filename):
    """Load a numpy array from a JSON document.

    Expected format:
    [
      {
        "observations": [0, 1]
      },
      {
        "observations": [0, 2]
      }    
    ]
    :param filename: Filename of JSON to load
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array."""
    return np.min(data, axis=0)


def daily_above_threshold(data, patient, threshold):
    """Determine the number of days on which the inflammation value exceeds a given threshold for
    a given patient.

    :param data: A 2D data array with inflammation data
    :param patient: The patient row number
    :param threshold: An inflammation threshold to check each daily value against
    :returns: The number of days on which the patient's daily inflammation exceeded the threshold
    """
    is_above_threshold = list(map(lambda x: x > threshold, data[patient]))
    return reduce(
        lambda sum_days_above, is_above: sum_days_above + 1 if is_above else sum_days_above,
        is_above_threshold,
        0
    )
