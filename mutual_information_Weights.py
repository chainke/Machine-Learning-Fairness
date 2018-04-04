import numpy as np
import Data_Generata
import math

from GLVQ.grlvq import GrlvqModel
from sklearn.metrics.cluster import normalized_mutual_info_score


#=======================================================
# Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information (MI) score to account for chance.
# It accounts for the fact that the MI is generally higher for two clusterings with a larger number of clusters,
# regardless of whether there is actually more information shared. For two clusterings U and V, the AMI is given as:
#
# AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [max(H(U), H(V)) - E(MI(U, V))]
#
# This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values
#  won’t change the score value in any way.
# =======================================================


def compute_mi_weights(feature, data):
    w = []
    for i in range(0, len(data[0])):
        temp_feature = []
        for j in range(0, len(data)):
            temp_feature.append(data[j][i])
        w.append(normalized_mutual_info_score(feature, temp_feature))
    return w


def compute_vector_weights(feature,labels,data):
    feature_mi = compute_mi_weights(feature,data)
    label_mi = compute_mi_weights(labels,data)
    relative_weights = []
    for i in range(0, len(feature_mi)):
        relative_weights.append(label_mi[i]*(1-feature_mi[i]))
    # idea: if the mutual information of a feature concerning the protected group is rather high, it should not be
    # treated with the same weight as unrelated features, since it might yield to discriminating results.
    # But if a feature also highly contributes to the label of the class, it should still have a major impact to not
    # worsen the results.

    print("Mutual information protected group -- features")
    for i in range(len(relative_weights)):
        print(str(feature_mi[i]))
    #print("---------------------------------------------------------")
    #print("protected feature:              " + str(feature_mi[0]))
    #print("number of children              " + str(feature_mi[1]))
    #print("annual income              " + str(feature_mi[2]))
    #print("time unemployed              " + str(feature_mi[3]))
    #print("married              " + str(feature_mi[4]))
    #print("=========================================================")
    print("Mutual information classification -- features")
    for i in range(len(relative_weights)):
        print(str(label_mi[i]))
    #print("---------------------------------------------------------")
    #print("protected feature              " + str(label_mi[0]))
    #print("number of children              " + str(label_mi[1]))
    #print("annual income              " + str(label_mi[2]))
    #print("time unemployed              " + str(label_mi[3]))
    #print("married              " + str(label_mi[4]))

    print ("weights: " + str(relative_weights));

    return relative_weights



def run_mi_data_generata():
    n = 1000#00
    p = 0.5
    k = 10

    weights = 'uniform'

    X, y = Data_Generata.CreditData(5,50000,10, 20000, 7,2).generate_credit_data(n,p,0.5)

    #print(X)

    feature = []
    for j in range(0, len(X)):
        feature.append(X[j][0])

    print(feature)

    compute_vector_weights(feature,y,X)


def grlvq_fit(X, y, feature):
    """
        Fits classifier for given data with given label. 
        Uses the grlvq model and the weights computed via mutial information

        Parameters
        ----------
        X: list of data as float vectors

        y: list of labels as int
           Either 0 or 1

        feature: list of protected feature as int
            Either 0 or 1

        Returns
        -------
        weights : list of float
            computed weights for features

        predY : list of
            predicted outcomes

    """

    #print(X)
    #print(feature)

    weights = compute_vector_weights(feature,y,X)

    #weights_processed = skip_smallest_weight(weights)
    #weights_processed = skip_weights_under_threshold(weights, 0.129)
    weights_processed = skip_weights_under_threshold(weights, 0.134)


    # start classification
    grlvq = GrlvqModel(initial_relevances = weights_processed)
    grlvq.fit(X, y)
    pred = grlvq.predict(X)

    print('classification accuracy:', grlvq.score(X, pred))
    return weights_processed, pred

def skip_smallest_weight(weights):
    smallest_weight = 1
    weight_index = 0

    # search for the smallest weight, except 0.0 since this is the protected feature
    for i in range(len(weights)):
        if(weights[i] != 0.0 and weights[i] < smallest_weight):
            smallest_weight = weights[i]
            weight_index = i

    print("eliminated smallest weight '%f' at position '%d'" % (smallest_weight, weight_index))

    weights_processed = weights
    weights_processed[weight_index] = 0.0

    print("weights: ",  str(weights_processed))

    return weights_processed



def skip_weights_under_threshold(weights, threshold):

    weights_processed = weights

    # eliminate all weights that are too small 
    for i in range(len(weights)):
        if(weights[i] != 0.0 and weights[i] < threshold):
            print("eliminated weight '%f' under threshold '%f' at position '%d'" % (weights_processed[i], threshold, i))
            weights_processed[i] = 0.0

    print("weights: ", str(weights_processed))

    return weights_processed