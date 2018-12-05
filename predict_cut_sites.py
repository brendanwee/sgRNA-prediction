from io_med import import_formatted_data
from reformat_data import split_x_y
from features import make_features
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error
from numpy import mean, var
from plotting import plot_scatter, plot_lines




def main(train_file, val_file, test_file, feature_file):
    # train
    data = import_formatted_data(train_file)
    X, y = split_x_y(data)
    transformed_X = make_features(X)

    with open(feature_file, "r") as f:
        features = [int(x) for x in f.readline().strip().split(" ")]


    selected_X = transformed_X[:, features]

    mod = SGDRegressor(max_iter=5000, penalty="l1")
    mod.fit(selected_X, y)
    train_preds = mod.predict(selected_X)

    val_data = import_formatted_data(val_file)
    val_X, val_y = split_x_y(val_data)
    transformed_val_X = make_features(val_X)
    selected_val_X = transformed_val_X[:,features]
    val_predictions = mod.predict(selected_val_X)

    test_data = import_formatted_data(test_file)
    test_X, test_y = split_x_y(test_data)
    transformed_test_X = make_features(test_X)
    selected_test_X = transformed_test_X[:, features]
    test_predictions = mod.predict(selected_test_X)

    print "Training"
    print "MSE:",mean_squared_error(y, train_preds)
    print "Mean, variance of real y:", mean(y), var(y)
    print "Mean, variance of pred y:", mean(train_preds), var(train_preds)

    print "Validation"
    print "MSE:", mean_squared_error(val_y, val_predictions)
    print "Mean, variance of real y:", mean(val_y), var(val_y)
    print "Mean, variance of Pred y:", mean(val_predictions), var(val_predictions)

    print "Test"
    print "MSE:", mean_squared_error(test_y, test_predictions)
    print "Mean, variance of real y:", mean(test_y), var(test_y)
    print "Mean, variance of Pred y:", mean(test_predictions), var(test_predictions)




main("train.tab", "val.tab", "small_test.tab", "selected_features_Lasso_alpha0.002.txt")