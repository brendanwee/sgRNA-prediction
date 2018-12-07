from io_med import import_raw_data_cutt_eff, write_data, import_formatted_data
from random import randint
from Bio import Align
from copy import deepcopy
import numpy as np
#name	seq	score	type

def reformat_data(filename):
    data = import_raw_data_cutt_eff(filename)
    new_data = []

    i = 0
    while i < len(data):
        name = data[i][0]
        assert data[i][3], "On-target sequence is not first sequence for target name, " + name
        guide_seq = data[i][1][:-3]
        while i < len(data) and name == data[i][0]:
            site_seq = data[i][1]
            cut_percent = data[i][2]
            new_data.append([site_seq, guide_seq, cut_percent])
            i += 1
    return new_data #


def flanking_pam(data):
    pam = True
    flanking = False
    for row in data:
        pam = pam and (row[1][-2:] == "GG")
        flanking = flanking or (len(row[0]) - len(row[1]) > 3)
    return pam, flanking


def find_guide_alignment(target, guide):
    aligner = Align.PairwiseAligner()
    aligner.query_internal_open_gap_score = -3
    aligner.target_internal_open_gap_score = -3
    aligner.target_left_open_gap_score = 0
    aligner.target_right_open_gap_score = 0
    aligner.query_left_open_gap_score = 0
    aligner.query_right_open_gap_score = 0
    aligner.match = 2

    alignments = aligner.align(target, guide)

    alignment = str(alignments[0])
    start = 0
    end = -1
    q = alignment[2 * len(target) + 2:-1]
    while q[start] == "-":
        start += 1
    while q[end] == "-":
        end -= 1
    end += 1
    return start, end


def format_target_guide(data):
    new_data = []
    pam,flanking = flanking_pam(data)
    for row in data:
        new_row = row
        if pam:
            new_row[1] = row[1][:-3]
        if flanking:
            start, end = find_guide_alignment(row[0], row[1]) # starting match, and last match index
            if end >-3:
                continue
            elif end == -3:
                new_row[0] = row[0][start:]
            else:
                new_row[0] = row[0][start:end + 3]
        new_data.append(new_row)
    return new_data


def shuffle_data(data):
    shuffled_data = []
    new_data = deepcopy(data)
    while new_data:
        i = randint(0,len(new_data)-1)
        shuffled_data.append(new_data.pop(i))
    return shuffled_data


def split_data(data, n):
    return data[:n], data[n:]


def split_train_val_test(data):
    num_train = len(data)*3/4
    num_val = len(data)/8
    return data[:num_train],data[num_train : num_train+num_val], data[num_train+num_val:]


def split_x_y(data):
    X = []
    y = []
    for row in data:
        X.append(row[:-1])
        y.append(row[-1])
    return np.array(X), np.array(y)

"""
data = import_formatted_data("test.tab")
val, test = split_data(data, 125)
write_data(val, "val.tab")
write_data(test, "small_test.tab")



Preprocessing steps - 

data = reformat_data("data/raw_data.tab")
write_data(data, "formatted_data.tab")
data = shuffle_data(data)
train, test = split_data(data, 250)
write_data(data, "shuffled_data.tab")
write_data(train, "train.tab")
write_data(test, "test.tab")
"""
