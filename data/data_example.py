import data.generator as generator
import data.gcd.process_csv_gcd as preprocess
import numpy as np
from sklearn_lvq.glvq import GlvqModel

# gen = generator.DataGen(verbose=False)
#
# std_array = np.array([0.2, 0.2, 0.2])
#
# X, C, Y = gen.generate_two_bubbles_multi_dim(number_data_points=1000, proportion_0=0.5, proportion_0_urban=0.8,
#                                              proportion_1_urban=0.5, proportion_0_pay=0.2, proportion_1_pay=0.5,
#                                              std=std_array)
#
# # Train a GLVQ model
# model = GlvqModel()
# model.fit(X, Y)
# Y_pred = model.predict(X)
#
# ax = generator.prepare_plot(X=X, C=C, Y=Y, Y_pred=Y_pred, prototypes=model.w_)
# generator.plot_prepared_dist(ax)

test_vector = np.array([1, 1, 2, 1, 3, 2, 1])[np.newaxis]

result = generator.normalize_category_feature(test_vector)
print(result.T)

# preprocess.process_gcd_to_csv("gcd/gcd.csv", "gcd/gcd_processed.csv")

original_data = preprocess.get_data("gcd/gcd.csv")

types = ["binary", "categories", "skip", "categories", "categories", "skip", "categories",
             "categories", "categories", "categories", "categories", "categories", "categories", "skip",
             "categories", "categories", "categories", "categories", "categories", "binary", "binary"]

result_data = preprocess.preprocess_data(original_data, types)

#gcd_data = preprocess.get_data("gcd/gcd_processed.csv")
# print(original_data)
#print(gcd_data.shape)
