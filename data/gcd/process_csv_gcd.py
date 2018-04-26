import csv
import numpy as np
import data.generator as generator


# processed_data_name = "dataset_processed.csv"

def get_data(file_path):
    """
    Returns the processed data as list.

    Parameters
    ----------
    file_path: string
        Path to file.

    Returns
    -------
    csv_data : np.array of floats
        Read file from .csv.

    """

    csv_data = []

    with open(file_path, newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in data_reader:
            csv_data.append(row)

    return np.array(csv_data)


def save_to_csv(data, path="result.csv"):
    print('Saving to CSV')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        # write rest of data
        writer.writerows(data)


def preprocess_data(data, feature_types):
    """
    Returns the processed data as list.

    Parameters
    ----------
    data: np.array
        Input data array.

    feature_types: list of int
        List of types of features:
        0: binary
        1: unnormalized
        2: categories
        3: skip

    Returns
    -------
    processed_data : np.array of floats
        Pre-processed data array.

    """

    num_rows, num_cols = data.shape

    processed_data = np.zeros(num_rows)[np.newaxis]
    for i in range(num_cols):
        col = data[:, i][np.newaxis]

        # print("type of col: {}".format(type(col)))

        if types[i] is "binary":
            processed_col = generator.normalize_binary_feature(col)
        elif types[i] is "unnormalized":
            processed_col = generator.normalize_feature(col)
        elif types[i] is "categories":
            processed_col = generator.normalize_category_feature(col)
        elif types[i] is "skip":
            continue

        #print("i: {}\ttype: {} \tprocessed_data: {} \t processed_col: {}".format(i, types[i], processed_data.shape,
        #                                                                         processed_col.shape))
        processed_data = np.concatenate((processed_data, processed_col), axis=0)

    return np.array(processed_data)


types = ["binary", "categories", "skip", "categories", "categories", "skip", "categories",
         "categories", "categories", "categories", "categories", "categories", "categories", "skip",
         "categories", "categories", "categories", "categories", "categories", "binary", "binary"]

read_data = get_data("gcd.csv")
proc_data = preprocess_data(read_data[1:], types)
save_to_csv(proc_data.T, path="gcd_processed.csv")
print(proc_data)
print(proc_data.shape)

# idea:
# load all data
# for each column
#  - read value
#  - translate/normalize
#  - add translated columns to output

# save output to new file (.csv/.npz)
