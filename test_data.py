import data.allbus.process_csv_allbus as allbus_data
import data.generator as generator
import numpy as np
from GLVQ.glvq import GlvqModel
import measures.functions as measure

import quad_fair_glvq as quad_glvq
import abs_fair_glvq as abs_glvq
import data.gcd.process_csv_gcd as gcd



#############################
# normalize the data and store it in csv file for a better overview
#############################

def test_allbus():
    # allbus
    X, y, protected, single_features = allbus_data.get_data()

    num_rows, num_cols = X.shape

    types = ["categories", "categories", "categories", "categories", "categories", "categories"]

    normalize_process = np.zeros(num_rows)[np.newaxis]
    for i in range(num_cols):
        col = X[:, i][np.newaxis]

        # print("type of col: {}".format(type(col)))
        if types[i] is "binary":
            processed_col = generator.normalize_binary_feature(col)
        elif types[i] is "unnormalized":
            processed_col = generator.normalize_feature(col)
        elif types[i] is "categories":
            processed_col = generator.normalize_category_feature(col)
        elif types[i] is "skip":
            continue

        #print("i: {}\ttype: {} \tprocessed_data: {} \t processed_col: {}".format(i, types[i], normalize_process.shape,
        #                                                                         processed_col.shape))
        normalize_process = np.concatenate((normalize_process, processed_col), axis=0)
        #X_normalized = normalize_process[1:].T
        X_normalized = normalize_process.T

    # write to csv for better overview
    allbus_data.write_to_csv(X_normalized, "normalized_data")

    # todo: probably convert data to float list, see gcd data


    #############################
    # compute fairness
    #############################
    print("\n\nfairness on allbus label: \n")
    measure.printAbsoluteMeasures(y.tolist(), protected.tolist())


    print("\n\nfairness on glvq label: \n")
    glvq = GlvqModel()
    glvq.fit(X_normalized,y)
    predicted_glvq = glvq.predict(X_normalized)

    #print(y.tolist())
    #print(predicted_glvq)
    print(predicted_glvq.tolist())

    measure.printAbsoluteMeasures(predicted_glvq.tolist(), protected.tolist())


def test_gcd():
    gcd_data = gcd.get_data("data/gcd/gcd_processed.csv")

    # split data in X, y and protected
    X = []
    y = []
    protected = []
    y_position = 1
    protected_position = len(gcd_data[0]) - 1
    skip_position = 0
    for i in range(len(gcd_data)):
        X_row = []
        for j in range(len(gcd_data[i])):
            if(j == y_position):
                y.append(float(gcd_data[i][j]))
            elif(j == protected_position):
                protected.append(float(gcd_data[i][j]))
            elif(j != skip_position):
                X_row.append(float(gcd_data[i][j]))
        X.append(X_row)

    print(X[0])

    print("\n\nfairness on gcd label: \n")
    measure.printAbsoluteMeasures(y, protected)

    print("\n\nfairness on glvq label: \n")
    glvq = GlvqModel()
    glvq.fit(X,y)
    glvq_predicted = glvq.predict(X)

    #print(glvq_predicted)

    print("\n\nfairness on abs_glvq label: \n")
    absglvq = abs_glvq.MeanDiffGlvqModel()
    absglvq.fit_fair(X,y,protected)
    absglvq_predicted = absglvq.predict(X)

    #print(absglvq_predicted)

    print("\n\nfairness on quad_glvq label: \n")
    quadglvq = quad_glvq.MeanDiffGlvqModel()
    quadglvq.fit_fair(X,y,protected)
    quadglvq_predicted = quadglvq.predict(X)

    #print(quadglvq_predicted)
    

test_gcd()


