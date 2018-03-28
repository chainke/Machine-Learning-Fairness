import numpy as np
import matplotlib.pyplot as plt

from Platt_Scaling_LVQ.glvq import GlvqModel
from Platt_Scaling_LVQ.plot_2d import to_tango_colors, tango_color

print(__doc__)

nb_ppc = 100
print('GLVQ:')

# generate random data
# TODO: use unfair data
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=nb_ppc),
    np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=nb_ppc), axis=0)
toy_label = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

# model fitting
# TODO: add platt scaling
glvq = GlvqModel()
glvq.fit(toy_data, toy_label)
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