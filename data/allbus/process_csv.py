import csv


def process_row(row):
	"""
	    Changes the real names of the features into for us usable integer values.
	    The mapping can be found in "value_dictionary".

	    Parameters
	    ----------
	    row: list of strings
	        the original names of the features

	    Returns
	    -------
	    new_row : list of strings
	        for us usable int values

	"""
	new_row = []
	for i in range(1,len(row)):
		new_row.append(value_dictionary[row[i]])

	return new_row


def get_data():
	"""
	Reads the csv and returns the necessary X, y and protected values.

	Returns
	-------
	X : list of int
	    allbus data with for us usable int values
	y : list of int, 0 or 1
		the labels for the data
	protected : list of int, 0 or 1
		whether the individual is in the protected group or not (currently gender, but this might be changed)

	"""

	X = []
	y = []
	protected = []

	label_position = 0
	protected_position = 1 # if it is gender

	with open(processed_data_name, newline = '') as csvfile:
		dataReader = csv.reader(csvfile, delimiter = ',', quotechar = '|')

		for row in dataReader:
			y.append(row[label_position])
			protected.append(row[protected_position])
			X.append(row[1:])

	return X, y, protected


processed_data = []
value_dictionary = {"BEFRISTET" : 0, "UNBEFRISTET" : 1, "FRAU" : 1, "MANN": 0, 
"MITTLERE REIFE": 0, "VOLKS-,HAUPTSCHULE": 1, "FACHHOCHSCHULREIFE" : 2, "HOCHSCHULREIFE" : 3, 
"OHNE ABSCHLUSS" : 4, "ANDERER ABSCHLUSS": 5, "JA": 1, "NEIN":0, "NA": 0,
"THUERINGEN" : 0, "EHEM. BERLIN-OST" : 1, "HESSEN" : 2, "HAMBURG" : 3, "NORDRHEIN-WESTFALEN": 4,
"SCHLESWIG-HOLSTEIN" : 5, "BRANDENBURG" : 6, "SACHSEN-ANHALT" : 7, "SACHSEN" : 8, "HAMBURG" : 9,
"BAYERN" : 10, "FRUEHERE DT.OSTGEB." : 11, "RHEINLAND-PFALZ" : 12, "NIEDERSACHSEN" : 13, "SONSTIGES LAND" : 14, 
"MECKLENB.-VORPOMMERN" : 15, "BADEN-WUERTTEMBERG" : 16, "SAARLAND" : 17, "BREMEN" : 18, "EHEM. BERLIN-WEST" : 19,
"UNTER 4 WOCHEN" : 1, "4 BIS 11 WOCHEN" : 2, "12 BIS 25 WOCHEN" : 3, "26 BIS 51 WOCHEN" : 4, "52 BIS 103 WOCHEN" : 5, 
"104 UND MEHR WOCHEN" : 6}
processed_data_name = "dataset_processed.csv"

# read and process data
with open('dataset_selected.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	# skip first row:
	next(readCSV, None)

	for row in readCSV:

		# only store those that don't have NA as label
		if(row[1] != "NA"):
			#print(row)
			new_row = process_row(row)
			#print(new_row)
			processed_data.append(new_row)


#print(str(processed_data))

with open(processed_data_name, 'w', newline = '') as csvfile:
	dataWriter = csv.writer(csvfile, delimiter=',')
	for i in range(len(processed_data)):
		dataWriter.writerow(processed_data[i])


#X,y,protected = get_data()
#print(y)
