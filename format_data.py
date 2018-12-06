from io_med import import_data
from reformat_data import split_train_val_test, shuffle_data
from features import make_features
from sys import argv


if __name__=="__main__":
    if len(argv) != 5:
        print """USAGE: format_data.py <target sequence index> <guide sequence index> <label sequence index>
        Writes four files name raw.tab, train.tab, val.tab, test.tab. Train, valm and test have expanded data with
        features, while raw will just contain the target, guide, and label."""
        exit()
    filename, target_i, guide_i, label_i = argv[1:5]#,argv[2],argv[3],argv[4]
    target_i, guide_i, label_i = int(target_i), int(guide_i), int(label_i)

    data = import_data(filename, target_i, guide_i, label_i) # Target, guide, label

    mix_data = shuffle_data(data)

    feature_data = make_features(mix_data)

    train, val, test = split_train_val_test(feature_data)

    with open("raw.tab", "w") as f:
        for row in data:
            entry = [str(x) for x in row]
            f.write("\t".join(entry)+"\n")

    with open("train.tab", "w") as f:
        for row in train:
            entry = [str(x) for x in row]
            f.write("\t".join(entry) + "\n")

    with open("val.tab", "w") as f:
        for row in val:
            entry = [str(x) for x in row]
            f.write("\t".join(entry) + "\n")

    with open("test.tab", "w") as f:
        for row in test:
            entry = [str(x) for x in row]
            f.write("\t".join(entry) + "\n")

