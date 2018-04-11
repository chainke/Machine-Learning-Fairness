import data.generator as generator
from sklearn_lvq.glvq import GlvqModel

gen = generator.DataGen()

X, C, Y = gen.generate_two_bubbles(number_data_points=1000, proportion_0=0.5, proportion_0_urban=0.8,
                                   proportion_1_urban=0.5, proportion_0_pay=0.8, proportion_1_pay=0.5)


# Train a GLVQ model
model = GlvqModel()
model.fit(X, Y)
Y_pred = model.predict(X)
# TODO: Find out why Y_pred is always true
print("Y_pred: {}".format(Y_pred))

ax = gen.prepare_plot(X=X, C=C, Y=Y, Y_pred=Y_pred)
gen.plot_prepared_dist(ax)
