import zliobaite
import numpy as np
from relaxed_equalized_odds.calib_eq_odds import Model
import pandas as pd

data = zliobaite.generate_data(100, 0.5, 0.1, 0.5)

print("new stuff")

for i in range(len(data)):
    if data[i][2] == 'F':
        data[i][2] = 1
    else:
        data[i][2] = 0

    print(data[i])

np.savetxt('data.csv', data, delimiter=',')

# Load the validation set scores from csvs
test_and_val_data = pd.read_csv('data.csv')

# Randomly split the data into two sets - one for computing the fairness constants
order = np.random.permutation(len(test_and_val_data))
val_indices = order[0::2]
test_indices = order[1::2]
val_data = test_and_val_data.iloc[val_indices]
test_data = test_and_val_data.iloc[test_indices]

# Create model objects - one for each group, validation and test
group_0_val_data = val_data[val_data['group'] == 0]
group_1_val_data = val_data[val_data['group'] == 1]
group_0_test_data = test_data[test_data['group'] == 0]
group_1_test_data = test_data[test_data['group'] == 1]

group_0_val_model = Model(group_0_val_data['prediction'].as_matrix(), group_0_val_data['label'].as_matrix())
group_1_val_model = Model(group_1_val_data['prediction'].as_matrix(), group_1_val_data['label'].as_matrix())
group_0_test_model = Model(group_0_test_data['prediction'].as_matrix(), group_0_test_data['label'].as_matrix())
group_1_test_model = Model(group_1_test_data['prediction'].as_matrix(), group_1_test_data['label'].as_matrix())

# Find mixing rates for equalized odds models
_, _, mix_rates = Model.eq_odds(group_0_val_model, group_1_val_model)

# Apply the mixing rates to the test models
eq_odds_group_0_test_model, eq_odds_group_1_test_model = Model.eq_odds(group_0_test_model,
                                                                       group_1_test_model,
                                                                       mix_rates)

# Print results on test model
print('Original group 0 model:\n%s\n' % repr(group_0_test_model))
print('Original group 1 model:\n%s\n' % repr(group_1_test_model))
print('Equalized odds group 0 model:\n%s\n' % repr(eq_odds_group_0_test_model))
print('Equalized odds group 1 model:\n%s\n' % repr(eq_odds_group_1_test_model))