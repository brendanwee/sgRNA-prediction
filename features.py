import numpy as np
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from Bio.SeqUtils import MeltingTemp as mt
from io_med import import_formatted_data
from reformat_data import split_x_y, format_target_guide

from numpy import argmin
from dicts import BASE_MAP
from joblib import dump
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from copy import deepcopy


def seq_to_int(seq):
    num = 0
    for i, base in enumerate(seq):
        num += BASE_MAP[base]*4**i
    return num


def hamming(x1,x2):
    return sum([a!=b for a,b in zip(x1,x2)])

def make_features_cut_eff(x_data):
    transformed_data = np.zeros((len(x_data), 884))

    for row_i  in range(0,len(x_data)):
        entry = x_data[row_i]

        # unpack
        seq = entry[0]
        guide = entry[1]

        # hot one encoding for seq - 23 *4 = 92
        for i in range(0, len(seq)):
            base = seq[i]
            if base == "-":
                i += 1
                continue
            transformed_data[row_i, 4*i+BASE_MAP[base]] = 1
            i += 1

        # one hot encoding for 2mer seq - 22 * 16 = 352; 352 + 92 = 444
        for i in range(0, len(seq) - 1):
            s = seq[i:i+2]
            if "-" in seq:
              continue
            seq_i = seq_to_int(s)
            transformed_data[row_i, 16*i+seq_i + 92] = 1

        # one hot encoding for guide - 20 * 4 = 80; 80 + 444 = 524
        for i in range(0, len(guide)):
            transformed_data[row_i, 444 + 4*i+BASE_MAP[guide[i]]] = 1


        # one hot encoding for 2mer guide - 19 * 16 = 304; 304 + 524 = 828
        for i in range(0, len(guide) - 1):
            guide_i = seq_to_int(guide[i:i+2])
            transformed_data[row_i, 16*i+guide_i + 524] = 1

        # melting temp
        rna_seq = Seq(guide[10:], generic_dna).transcribe()
        complement_DNA = Seq(seq[10:-3], generic_dna).complement()


        # NN Thermodynamics for 3' end
        try:
            transformed_data[row_i, 829] = mt.Tm_NN(rna_seq, c_seq=complement_DNA,
                nn_table=mt.R_DNA_NN1, de_table=mt.DNA_DE1)
        except ValueError:
            pass

        # num_mismatch
        assert len(seq[:-3]) == len(guide)
        transformed_data[row_i, 829] = hamming(seq[:-3], guide) # 830

        # one hot encoding of mismatches in guide - 20 -> 850
        for i in range(0, len(guide)):
            if guide[i] != seq[i]:
                transformed_data[row_i, 830+i] = 1

        # PAM dinucleotides = 32
        index = seq_to_int(seq[-3:-1])
        transformed_data[row_i, index + 850] = 1
        index = seq_to_int(seq[-2:])
        transformed_data[row_i, index + 866] = 1


        GC = sum([1 for x in guide if x in ("G","C")])
        AT = sum([1 for x in guide if x in ("A","T")])
        if GC - AT > 10:
            transformed_data[row_i, 882] = 1 # GC high
        elif AT - GC > 10:
            transformed_data[row_i, 883] = 1 # GC Low

    return transformed_data

def make_features(data):
    transformed_data = np.zeros((len(data), 885))
    data = format_target_guide(data)

    for row_i in range(0,len(data)):
        entry, y = data[row_i][:-1], data[row_i][-1]

        # unpack
        seq = entry[0]
        guide = entry[1]

        # hot one encoding for seq - 23 *4 = 92
        for i in range(0, len(seq)):
            base = seq[i]
            if base == "-":
                i += 1
                continue
            transformed_data[row_i, 4*i+BASE_MAP[base]] = 1
            i += 1

        # one hot encoding for 2mer seq - 22 * 16 = 352; 352 + 92 = 444
        for i in range(0, len(seq) - 1):
            s = seq[i:i+2]
            if "-" in seq:
              continue
            seq_i = seq_to_int(s)
            transformed_data[row_i, 16*i+seq_i + 92] = 1

        # one hot encoding for guide - 20 * 4 = 80; 80 + 444 = 524
        for i in range(0, len(guide)):
            transformed_data[row_i, 444 + 4*i+BASE_MAP[guide[i]]] = 1


        # one hot encoding for 2mer guide - 19 * 16 = 304; 304 + 524 = 828
        for i in range(0, len(guide) - 1):
            guide_i = seq_to_int(guide[i:i+2])
            transformed_data[row_i, 16*i+guide_i + 524] = 1

        # melting temp
        rna_seq = Seq(guide[10:], generic_dna).transcribe()
        complement_DNA = Seq(seq[10:-3], generic_dna).complement()


        # NN Thermodynamics for 3' end
        try:
            transformed_data[row_i, 829] = mt.Tm_NN(rna_seq, c_seq=complement_DNA,
                nn_table=mt.R_DNA_NN1, de_table=mt.DNA_DE1)
        except ValueError:
            pass

        # num_mismatch
        if len(seq[:-3]) != len(guide):
            if seq == "":
                continue
        transformed_data[row_i, 829] = hamming(seq[:-3], guide) # 830

        # one hot encoding of mismatches in guide - 20 -> 850
        for i in range(0, len(guide)):
            if i == len(seq):
                print "here?"
                print data[row_i]
                print seq
                print guide
                exit()
            if guide[i] != seq[i]:
                transformed_data[row_i, 830+i] = 1

        # PAM dinucleotides = 32
        index = seq_to_int(seq[-3:-1])
        transformed_data[row_i, index + 850] = 1
        index = seq_to_int(seq[-2:])
        transformed_data[row_i, index + 866] = 1


        GC = sum([1 for x in guide if x in ("G","C")])
        AT = sum([1 for x in guide if x in ("A","T")])
        if GC - AT > 10:
            transformed_data[row_i, 882] = 1 # GC high
        elif AT - GC > 10:
            transformed_data[row_i, 883] = 1 # GC Low
        transformed_data[row_i, 884] = y

    return transformed_data


def select_features_and_model(train_x, train_y, val_x, val_y, dataset_name):

    feature_selectors = [
                         SelectFromModel(DecisionTreeRegressor(), max_features=50),
                         SelectFromModel(DecisionTreeRegressor(), max_features=75),
                         SelectFromModel(DecisionTreeRegressor(), max_features=100),

                         SelectFromModel(ExtraTreesRegressor(),max_features=50),
                         SelectFromModel(ExtraTreesRegressor(), max_features=75),
                         SelectFromModel(ExtraTreesRegressor(), max_features=100),

                         SelectFromModel(RandomForestRegressor(), max_features=50),
                         SelectFromModel(RandomForestRegressor(), max_features=75),
                         SelectFromModel(RandomForestRegressor(), max_features=100)]

    results = []

    regressors = [ExtraTreesRegressor(),
                  RandomForestRegressor(),
                  AdaBoostRegressor(),
                  GradientBoostingRegressor(),
                  GradientBoostingRegressor(loss='lad'),
                  GradientBoostingRegressor(loss='huber'),
                  SVR(max_iter=5000),


                  ]

    # SGDRegressor(max_iter=5000, penalty="l1"),
    #GaussianProcessRegressor()

    feature_selector_strings = [

                                "SelectFromModel(DecisionTreeRegressor(), max_features=50)",
                                "SelectFromModel(DecisionTreeRegressor(), max_features=75)",
                                "SelectFromModel(DecisionTreeRegressor(), max_features=100)",

                                "SelectFromModel(ExtraTreesRegressor(), max_features=50)",
                                "SelectFromModel(ExtraTreesRegressor(), max_features=75)",
                                "SelectFromModel(ExtraTreesRegressor(), max_features=100)",

                                "SelectFromModel(RandomForestRegressor(), max_features=50)",
                                "SelectFromModel(RandomForestRegressor(), max_features=75)",
                                "SelectFromModel(RandomForestRegressor(), max_features=100)"]

    regressor_strings = ["ExtraTreesRegressor()",
                         "RandomForestRegressor()",
                         "AdaBoostRegressor()",
                         "GradientBoostingRegressor()",
                         "GradientBoostingRegressor(loss='lad')",
                         "GradientBoostingRegressor(loss='huber')",
                         "SVR()",

                         ]
    "SGDRegressor(max_iter=5000, penalty='l1')"
    "GaussianProcessRegressor()"

    for selector, selector_s in zip(feature_selectors, feature_selector_strings):  # iterates across both at same time
        print "selecting with ", selector_s
        selector.fit(train_x, train_y)
        features = selector.get_support(indices=True)

        selected_train_x = train_x[:, features]
        selected_val_x = val_x[:, features]
        errors = []

        for model, model_s in zip(regressors, regressor_strings):
            print "predicting with ", model_s
            model.fit(selected_train_x, train_y)

            predictions = model.predict(selected_val_x)

            error = mean_squared_error(val_y, predictions)
            errors.append((error, deepcopy(model), features))
        results.append(errors)
    with open("log.txt", "w") as f:
        best = (0, 0, model, 99999, [])  # selector index, regressor index, model, error, features
        for selector_i, errors in enumerate(results):
            for regressor_i, model_stats in enumerate(errors):
                error = model_stats[0]
                model = model_stats[1]
                features = model_stats[2]

                selector_str = feature_selector_strings[selector_i]
                regressor_str = regressor_strings[regressor_i]

                f.write("validation error for  " + selector_str + "  feature selector and  " + regressor_str + "  is  " + str(error))

                if error < best[3]:
                    best = (selector_i, regressor_i, model, error, features)

    featureselector_str = feature_selector_strings[best[0]]
    model_str = regressor_strings[best[1]]
    b_model = best[2]
    error = best[3]
    feats = best[4]

    dump(b_model, dataset_name+".joblib")

    with open(dataset_name+"_features.txt", "w") as f:
        feats = [str(x) for x in feats]
        f.write("\t".join(feats))

    with open(dataset_name + "_performance.txt", "w") as f:
        f.write("validation error for " + featureselector_str + " feature selector and " + model_str + " is " + str(error))
    return b_model, feats


def select_and_plot_features(train_file, val_file, test_file):
    # train
    data = import_formatted_data(train_file)
    X, y = split_x_y(data)
    transformed_X = make_features(X)

    alphas = [.0001,.0003,.0005,.0007, .001,.002,.003, .005, .01]
    nums = []
    train_mse = []
    val_mse = []

    # TODO: build list of feature selectors from scikit learn. Ex. SVM, ReliefF

    for a in alphas:
        mod = Lasso(alpha=a, max_iter=10000)
        mod.fit(transformed_X, y)
        params = mod.coef_
        num_features = sum([1 for x in params if x != 0])
        nums.append(num_features)

        features = [i for i,x in enumerate(params) if x != 0]

        """with open("selected_features_Lasso_alpha" + str(a) + ".txt", "w") as f:
            f.write(" ".join([str(x) for x in features]))"""

        selected_X = transformed_X[:, features]

        mod = SGDRegressor(max_iter=5000, penalty="l1")
        mod.fit(selected_X, y)
        train_preds = mod.predict(selected_X)

        val_data = import_formatted_data(val_file)
        val_X, val_y = split_x_y(val_data)
        transformed_test_X = make_features(val_X)
        selected_val_X = transformed_test_X[:,features]
        predictions = mod.predict(selected_val_X)
        train_mse.append(mean_squared_error(y, train_preds))
        #print mean(y), var(y)
        #print mean(train_preds), var(train_preds)
        val_mse.append(mean_squared_error(val_y, predictions))

        #print mean(val_y), var(val_y)
        #print mean(predictions), var(predictions)
    print argmin(val_mse)
    #plot_lines(nums, "num_features", "Features_vs_MSE", train_mse, "Training_MSE", val_mse, "Val_MSE")

#select_and_plot_features("train.tab", "val.tab", "small_test.tab")