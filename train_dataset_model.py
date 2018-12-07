#!/anaconda/bin/python

from sys import argv
from features import select_features_and_model
from io_med import import_feature_data
from reformat_data import split_x_y

if __name__=="__main__":
    if len(argv) != 4:
        print """USAGE: train_dataset_model.py <traindata> <valdata> <out-prefix>
        Reads in the data from both files and runs feature selection and model selection pipeline"""

    trainfile = argv[1]
    valfile = argv[2]
    datasetname = argv[3]

    train_data = import_feature_data(trainfile)
    val_data = import_feature_data(valfile)

    train_x, train_y = split_x_y(train_data)
    val_x, val_y = split_x_y(val_data)

    select_features_and_model(train_x, train_y, val_x, val_y, datasetname)
