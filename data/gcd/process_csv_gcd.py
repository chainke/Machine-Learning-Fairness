import csv
import numpy as np
import data.generator as generator


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
    """
    Saves data matrix to a new csv-file.

    Parameters
    ----------
    data: np.array
        Data matrix to write.

    path: string
        Path to output file.

    """

    print('Saving to CSV')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        # write rest of data
        writer.writerows(data)


def preprocess_data(data, feature_types, verbose=False):
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
    print("initial preprocessed_data: {}".format(processed_data))
    for i in range(num_cols):
        col = data[:, i][np.newaxis]
        print(col)
        print("type of col: {}".format(type(col)))

        if feature_types[i] is "binary":
            processed_col = generator.normalize_binary_feature(col)
        elif feature_types[i] is "unnormalized":
            processed_col = generator.normalize_feature(col)
        elif feature_types[i] is "categories":
            processed_col = generator.normalize_category_feature(col)
        elif feature_types[i] is "metric":
            processed_col = generator.normalize_metric_feature(col)
        elif feature_types[i] is "skip":
            continue

        print("current processed_col: {}".format(processed_col))

        if verbose:
            print("i: {}\ttype: {} \tprocessed_data: {} \t processed_col: {}".format(i, feature_types[i],
                                                                                     processed_data.shape,
                                                                                     processed_col.shape))
        processed_data = np.concatenate((processed_data, processed_col), axis=0)
        print("preprocessed_data of {}: {}".format(i, processed_data))
    return np.array(processed_data)


def process_gcd_to_csv(in_path="gcd.csv", out_path="gcd_processed.csv"):
    """
    Reads gcd data and writes it to a new csv-file.

    Parameters
    ----------
    in_path: string
        Path to input file.

    out_path: string
        Path to input file.

    """
    read_data = get_data(in_path)

    types = ["binary", "categories", "metric", "categories", "categories", "metric", "categories",
             "categories", "categories", "categories", "categories", "categories", "categories", "metric",
             "categories", "categories", "categories", "categories", "categories", "binary", "binary"]

    proc_data = preprocess_data(read_data[1:], types)
    save_to_csv(proc_data.T, path=out_path)
