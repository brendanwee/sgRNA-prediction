from features import  select_features_and_model
from sklearn.metrics import mean_squared_error
import numpy as np
from joblib import load
from io_med import import_train_val_test_data
from reformat_data import split_x_y
from compile_all_data_and_normalize import compile_and_normalize
from reformat_data import shuffle_data, split_train_val_test
from features import make_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


def add_model_feature(model, features, x):
    preds = model.predict(x[:,features])

    print x.shape
    new_x = np.append(x,np.array(np.matrix(preds).transpose()), 1)
    print new_x.shape
    print ""

    return new_x


def get_features(filename):
    with open(filename, 'r') as f:
        features = f.readline().strip().split("\t")
    return [int(x) for x in features]


def main():

    all_data = compile_and_normalize()

    mix_data = shuffle_data(all_data)

    feature_data = make_features(mix_data)

    train_data, val_data, test_data = split_train_val_test(feature_data)


    modelfiles = [
        "data/CRISPOR_readFraction_off_target/CRISPOR_readFraction_off_target.joblib",
        "data/Azimuth/Azimuth.joblib",
        "data/Res6tg/Res6tg.joblib",
        "data/Rule_set_1_log2change_on_target/Rule_set_1_log2change_on_target.joblib"
    ]

    featurefiles = [
        "data/CRISPOR_readFraction_off_target/CRISPOR_readFraction_off_target_features.txt",
        "data/Azimuth/Azimuth_features.txt",
        "data/Res6tg/Res6tg_features.txt",
        "data/Rule_set_1_log2change_on_target/Rule_set_1_log2change_on_target_features.txt"
    ]

    ensemble = []

    for modelfile, featurefile in zip(modelfiles, featurefiles):
        print modelfile
        print featurefile
        model = load(modelfile)

        features = get_features(featurefile)

        ensemble.append((model,features))



    train_x, train_y = split_x_y(train_data)
    val_x, val_y = split_x_y(val_data)
    test_x, test_y = split_x_y(test_data)

    featureselector = SelectFromModel(RandomForestRegressor(), max_features=100)
    regressor = RandomForestRegressor()

    featureselector.fit(train_x, train_y)
    features = featureselector.get_support(indices=True)

    selected_train_x = train_x[:, features]
    selected_val_x = val_x[:, features]
    selected_test_x = test_x[:,features]

    regressor.fit(selected_train_x, train_y)

    train_predictions = regressor.predict(selected_train_x)
    val_predictions = regressor.predict(selected_val_x)
    test_predictions = regressor.predict(selected_test_x)

    train_error = mean_squared_error(train_y, train_predictions)
    validataion_error = mean_squared_error(val_y, val_predictions)
    test_error = mean_squared_error(test_y, test_predictions)

    for model, features in ensemble:
        add_train_x = add_model_feature(model, features, train_x)
        add_val_x = add_model_feature(model, features, val_x)
        add_test_x = add_model_feature(model, features, test_x)

    #model, features = select_features_and_model(add_train_x, train_y, add_val_x, val_y,"all_data_toplayer")

    modelfile = "all_data_toplayer.joblib"
    featurefile = "all_data_toplayer_features.txt"
    model = load(modelfile)
    features = get_features(featurefile)

    train_pred = model.predict(add_train_x[:,features])
    val_pred = model.predict(add_val_x[:,features])
    test_pred = model.predict(add_test_x[:,features])

    train_MSE = mean_squared_error(train_y, train_pred)
    val_MSE = mean_squared_error(val_y, val_pred)
    test_MSE = mean_squared_error(test_y, test_pred)

    with open("all_data_top_layer_MSE.txt", "w") as f:
        f.write("ensemble train MSE: " + str(train_MSE) + "\n")
        f.write("ensemble val MSE: " + str(val_MSE) + "\n")
        f.write("ensemble test MSE: " + str(test_MSE) + "\n")
        f.write("train MSE: " + str(train_error) + "\n")
        f.write("val MSE: " + str(validataion_error) + "\n")
        f.write("test MSE: " + str(test_error) + "\n")


main()
