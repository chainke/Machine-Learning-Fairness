import zliobaite
import relaxed_equalized_odds.calib_eq_odds as odds
from sklearn.model_selection import train_test_split
import numpy as np

data = zliobaite.generate_data(100, 0.5, 0.1, 0.5)

prediction = 0
label = 1
group = 2

savedata = [['', 'label', 'group', 'prediction']]

print("--------------- start testfile ---------------")

for i in range(len(data)):
    if data[i][2] == 'F':
        savedata.append([i, int(data[i][1]), 1, data[i][0]])
    else:
        savedata.append([i, int(data[i][1]), 0, data[i][0]])

    print(savedata[i])

np.savetxt('relaxed_equalized_odds/data/data.csv', savedata, delimiter=',', fmt='%s')

# test_data, val_data, _, _ = train_test_split(data, data, test_size=1/2, random_state=42)
#
# # Create model objects - one for each group, validation and test
# group_0_val_data = val_data[val_data[group] == 0]
# group_1_val_data = val_data[val_data[group] == 1]
# group_0_test_data = test_data[test_data[group] == 0]
# group_1_test_data = test_data[test_data[group] == 1]
#
# group_0_val_model = odds.Model(group_0_val_data[prediction], group_0_val_data[label])
# group_1_val_model = odds.Model(group_1_val_data[prediction], group_1_val_data[label])
# group_0_test_model = odds.Model(group_0_test_data[prediction], group_0_test_data[label])
# group_1_test_model = odds.Model(group_1_test_data[prediction], group_1_test_data[label])
#
# # Find mixing rates for equalized odds models
# _, _, mix_rates = odds.Model.eq_odds(group_0_val_model, group_1_val_model)
#
# # Apply the mixing rates to the test models
# eq_odds_group_0_test_model, eq_odds_group_1_test_model = odds.Model.eq_odds(group_0_test_model,
#                                                                        group_1_test_model,
#                                                                        mix_rates)
#
# # Print results on test model
# print('Original group 0 model:\n%s\n' % repr(group_0_test_model))
# print('Original group 1 model:\n%s\n' % repr(group_1_test_model))
# print('Equalized odds group 0 model:\n%s\n' % repr(eq_odds_group_0_test_model))
# print('Equalized odds group 1 model:\n%s\n' % repr(eq_odds_group_1_test_model))