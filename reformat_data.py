from io_med import import_raw_data, write_data, import_formatted_data
from random import randint

#name	seq	score	type

def reformat_data(filename):
    data = import_raw_data(filename)
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
    return new_data


def shuffle_data(data):
    shuffled_data = []
    while data:
        i = randint(0,len(data)-1)
        shuffled_data.append(data[i])
        del data[i]
    return shuffled_data


def split_data(data, n):
    return data[:n], data[n:]


def split_x_y(data):
    X = []
    y = []
    for row in data:
        X.append(row[:-1])
        y.append(row[-1])
    return X, y

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
