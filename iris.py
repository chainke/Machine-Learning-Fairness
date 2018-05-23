from sklearn import datasets
import data.generator as generator
import numpy as np
import measures.functions as measure
from GLVQ.glvq import GlvqModel

import quad_fair_glvq as quad_glvq
import abs_fair_glvq as abs_glvq
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# import some data to play with
basic_iris = datasets.load_iris()
iris = np.transpose(np.array(basic_iris.data))
norm_iris = [(generator.normalize_metric_feature(np.array(x)[np.newaxis])[0]) for x in iris]

X = np.transpose(norm_iris[:3])
y = [0 if x >= 2 else 1 for x in basic_iris.target]
# y = basic_iris.target.tolist()

protected = [1 if x >= 0.5 else 0 for x in norm_iris[3]]

train_X, test_X, train_y, test_y, train_protected, test_protected = train_test_split(X, y, protected, test_size=0.33, random_state=42)




set_alpha = 10

print("\n\nfairness on gcd label: \n")
measure.printAbsoluteMeasures(test_y, test_protected)

print("\n\nfairness on glvq label: \n")
glvq = GlvqModel()
glvq.fit(train_X, train_y)
glvq_predicted = glvq.predict(test_X)
measure.printAbsoluteMeasures(glvq_predicted.tolist(), test_protected)


print("\n\nfairness on abs_glvq label: \n")
absglvq = abs_glvq.MeanDiffGlvqModel(alpha=set_alpha)
absglvq.fit_fair(train_X, train_y, train_protected)
absglvq_predicted = absglvq.predict(test_X)
measure.printAbsoluteMeasures(absglvq_predicted.tolist(), test_protected)


print("\n\nfairness on quad_glvq label: \n")
quadglvq = quad_glvq.MeanDiffGlvqModel(alpha=set_alpha)
quadglvq.fit_fair(train_X, train_y, train_protected)
quadglvq_predicted = quadglvq.predict(test_X)
measure.printAbsoluteMeasures(quadglvq_predicted.tolist(), test_protected)

print("accuracy unfair:", accuracy_score(test_y, glvq_predicted))
print("accuracy abs:", accuracy_score(test_y, absglvq_predicted))
print("accuracy quad:", accuracy_score(test_y, quadglvq_predicted))

print("predict unfair:", glvq_predicted)
print("predict abs:", absglvq_predicted)
print("predict quad:", quadglvq_predicted)
