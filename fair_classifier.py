import numpy as np
import matplotlib.pyplot as plt
import Data_Generata

from fair_glvq import GlvqModel
from GLVQ.plot_2d import to_tango_colors, tango_color


def split_x(x, dim_protected):
    protected = []
    new_x = []

    for i in range(0, len(x)):
        protected.append(x[i][dim_protected])
        new_x.append(
            x[i][:dim_protected] + x[i][dim_protected + 1:]
        )

    return new_x, protected


print(__doc__)

nb_ppc = 100
print('Fair GLVQ:')

# generate random data
# TODO: use unfair data
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=nb_ppc),
    np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=nb_ppc),axis=0)
toy_label = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)
toy_protected_labels = np.append(np.zeros(nb_ppc), np.ones(nb_ppc))  #np.random.randint(2, size=2*nb_ppc)
print(len(toy_protected_labels))
print(len(toy_data))

weights = 'uniform'

# model fitting
# TODO: add platt scaling
glvq = GlvqModel(0.5)
# glvq.fit(new_x, y, protected_labels)
glvq.fit(toy_data, toy_label, toy_protected_labels)
pred = glvq.predict(toy_data)

# plotting
plt.scatter(toy_data[:, 0], toy_data[:, 1], c=to_tango_colors(toy_label), alpha=0.5)
plt.scatter(toy_data[:, 0], toy_data[:, 1], c=to_tango_colors(pred), marker='.')
plt.scatter(glvq.w_[:, 0], glvq.w_[:, 1],
            c=tango_color('aluminium', 5), marker='D')
plt.scatter(glvq.w_[:, 0], glvq.w_[:, 1],
            c=to_tango_colors(glvq.c_w_, 0), marker='.')
plt.axis('equal')

print('classification accuracy:', glvq.score(toy_data, toy_label))
plt.show()