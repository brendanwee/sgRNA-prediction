from io_med import import_feature_data
from reformat_data import split_x_y
from features_bren import select_features_and_model

if __name__ == "__main__":
    dataset_name = "CRISPOR_readFraction_off_target"
    train_data = import_feature_data("data/CRISPOR_readFraction_off_target/train.tab")
    val_data = import_feature_data("data/CRISPOR_readFraction_off_target/val.tab")

    train_x, train_y = split_x_y(train_data)
    val_x, val_y = split_x_y(val_data)


    select_features_and_model(train_x, train_y, val_x, val_y, dataset_name)