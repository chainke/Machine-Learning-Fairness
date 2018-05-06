import data.generator as generator
import data.gcd.process_csv_gcd as preprocess
import numpy as np
from sklearn_lvq.glvq import GlvqModel

#
# ------------------------------------
# Train a GLVQ model for bubble data
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



# ------------------------------------
# preprocess gcd data
# test_vector = np.array([1, 1, 2, 1, 3, 2, 1])[np.newaxis]
#
# result = generator.normalize_category_feature(test_vector)
# print(result.T)
#
# # preprocess.process_gcd_to_csv("gcd/gcd.csv", "gcd/gcd_processed.csv")
#
# original_data = preprocess.get_data("gcd/gcd.csv")
#
# types = ["binary", "categories", "skip", "categories", "categories", "skip", "categories",
#              "categories", "categories", "categories", "categories", "categories", "categories", "skip",
#              "categories", "categories", "categories", "categories", "categories", "binary", "binary"]
#
# result_data = preprocess.preprocess_data(original_data, types)
#
# #gcd_data = preprocess.get_data("gcd/gcd_processed.csv")
# # print(original_data)
# #print(gcd_data.shape)


# ------------------------------------
# Train a GLVQ model for Japanese data

jap_data = preprocess.get_data("../relaxed_equalized_odds/data/japanese_screening.csv")[1:]

processed_jap_data = np.array([(x[1:2] + x[2+2:]) for x in jap_data]).astype(np.float)

normalized_processed_jap_data = []
n, m = processed_jap_data.shape
for i in range(m):
    normalized_processed_jap_data.append(generator.normalize_metric_feature(np.array([x[i] for x in processed_jap_data])[np.newaxis]))

normalized_processed_jap_data = zip(*normalized_processed_jap_data)

print(normalized_processed_jap_data)

#preprocess.preprocess_data(processed_jap_data, [""])

# print(normalized_processed_jap_data.shape)

plot_jap_data = np.array([(x[5:7]) for x in normalized_processed_jap_data])

y = np.array([x[0] for x in jap_data]).astype(np.float)
# print("y: {}".format(y))
c = np.array([x[3] for x in jap_data]).astype(np.float)
# print("c: {}".format(c))


model = GlvqModel()
model.fit(processed_jap_data, y)

Y_pred = model.predict(processed_jap_data)

print("Y_pred: {}".format(Y_pred))

ax = generator.prepare_plot(X=plot_jap_data.T, C=c, Y=y, Y_pred=Y_pred, prototypes=model.w_, title="Japanese Data",
                            label1="month until payback", label2="years_at_work")
generator.plot_prepared_dist(ax)
