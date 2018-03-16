import numpy as np
import Data_Generata
import math


from sklearn.metrics.cluster import adjusted_mutual_info_score


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
        w.append(adjusted_mutual_info_score(feature, temp_feature))
    return w


def compute_vector_weights(feature,labels,data):
    feature_mi = compute_mi_weights(feature,data)
    label_mi = compute_mi_weights(labels,data)

    # idea: if the mutual information of a feature concerning the protected group is rather high, it should not be
    # treated with the same weight as unrelated features, since it might yield to discriminating results.
    # But if a feature also highly contributes to the label of the class, it should still have a major impact to not
    # worsen the results.

    print("Mutual information protected group -- features")
    print("---------------------------------------------------------")
    print("protected feature:              " + str(feature_mi[0]))
    print("number of children              " + str(feature_mi[1]))
    print("annual income              " + str(feature_mi[2]))
    print("time unemployed              " + str(feature_mi[3]))
    print("married              " + str(feature_mi[4]))
    print("=========================================================")
    print("Mutual information classification -- features")
    print("---------------------------------------------------------")
    print("protected feature              " + str(label_mi[0]))
    print("number of children              " + str(label_mi[1]))
    print("annual income              " + str(label_mi[2]))
    print("time unemployed              " + str(label_mi[3]))
    print("married              " + str(label_mi[4]))



n = 100
p = 0.5
k = 10

weights = 'uniform'

X, y = Data_Generata.CreditData(5,50000,10, 20000, 7,2).generate_credit_data(n,p)

feature = []
for j in range(0, len(X)):
    feature.append(X[j][0])

compute_vector_weights(feature,y,X)