from dicts import INT_TO_BASE
from random import randint
import numpy as np

def import_raw_data_cutt_eff(filename):
    data = []
    with open(filename, "r") as f:
        line = f.readline().strip()
        while line != "":
            row = line.split("\t")
            if row[2] != "NA":
                row[2] = float(row[2])
            if row[3] == "on-target":
                row[3] = 1
            else:
                row[3] = 0
            data.append(row)
            line = f.readline().strip()
    return data


def import_azimuth_data(filename, target_i, guide_i, label_i):
    data = []
    with open(filename, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            try:
                float(row[label_i]) # header?
            except ValueError:
                continue
            target = row[target_i]
            if row[-1] == "PAM":
                target = target + row[3]
            else:
                N = INT_TO_BASE[randint(0,3)]
                target = target + N + "GG"
            new_row = [target, row[guide_i], float(row[label_i])]
            data.append(new_row)
    return data


def import_data(filename, target_i, guide_i, label_i):
    data = []
    with open(filename, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            try:
                float(row[label_i]) # header?
            except ValueError:
                continue

            new_row = [row[target_i], row[guide_i], float(row[label_i])]
            data.append(new_row)
    return data


def import_formatted_data(filename):
    data = []
    with open(filename, "r") as f:
        line = f.readline().strip()
        while line != "":
            row = line.split("\t")
            if row[2] == "NA":
                line = f.readline().strip()
                continue
            row[2] = float(row[2])
            data.append(row)
            line = f.readline().strip()
    return data


def import_feature_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            row = [float(x) for x in row]
            data.append(row)
    return np.array(data)


def write_data(data, filename):
    with open(filename, "w") as f:
        for row in data:
            row = [str(x) for x in row]
            f.write("\t".join(row) + "\n")


def import_train_val_test_data(dirname):
    trainfile = "data/" + dirname + "/train.tab"
    valfile = "data/" + dirname + "/val.tab"
    testfile = "data/" + dirname + "/test.tab"

    train_data = import_feature_data(trainfile)
    val_data = import_feature_data(valfile)
    test_data = import_feature_data(testfile)

    return train_data, val_data, test_data

