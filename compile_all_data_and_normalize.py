from __future__ import division
from io_med import import_formatted_data
import numpy as np


def normalize_data(data):
    data = np.array(data)
    labels = np.array([float(x) for x in data[:,-1]])
    lowest = min(labels)
    if lowest < 0:
        labels = labels + abs(lowest)
    highest = max(labels)
    labels = labels/float(highest)
    data[:,-1] = labels
    return data



def compile_and_normalize():
    compiled = []

    filename = "data/CRISPOR_readFraction_off_target/raw.tab"
    data = import_formatted_data(filename)
    data = np.array(data)
    labels = np.array([float(x) for x in data[:,-1]])
    labels = labels * -1
    data[:,-1] = labels

    normalized_data = normalize_data(data)
    for row in normalized_data:
        compiled.append(row)

    files = [
        "data/Azimuth/raw.tab",
        "data/Res6tg/raw.tab",
        "data/Rule_set_1_log2change_on_target/raw.tab"
        ]


    for filename in files:
        data = import_formatted_data(filename)
        normalized_data = normalize_data(data)
        for row in normalized_data:
            compiled.append(row)
    return compiled
