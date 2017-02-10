import csv
import numpy as np
import sys
import math

def read_csv(filename, columns):
	quotes_stripper = lambda x: x.strip('"').strip(" ")
	converter_all_cols = {x:quotes_stripper for x in columns}
	return np.genfromtxt(filename, dtype="string", delimiter=",", skip_header=1, usecols=columns, autostrip=True, converters = converter_all_cols)

def min_max_normalize(data, column, new_min, new_max):
	data_arr = data
	column_values = data_arr[:, column]
	current_min = min(column_values)
	current_max = max(column_values)
	old_range = current_max - current_min
	new_range = new_max - new_min
	normalize_func = lambda x: new_min + (x-current_min)*new_range/(old_range) if not np.isnan(x) else np.nan
	return np.vectorize(normalize_func)(column_values)

def minkowski_distance(vector, another_vector, r):
	distance = np.vectorize(lambda x: pow(x, r))
	return pow(sum(distance(abs(vector - another_vector))), 1.0/r)

def euclidean_distance(vector, another_vector):
	return minkowski_distance(vector, another_vector, 2)

def distance_to_similarity(distance_value):
	return 1.0/(1+distance_value)

def magnitude(vector):
	return pow(sum(pow(vector, 2)), 0.5)

def cosine_similarity(vector, another_vector):
	return sum(vector * another_vector)/(magnitude(vector)*magnitude(another_vector))

def normalize_data(records, columns):
	for i in columns:
		records[:, i] = min_max_normalize(records, i, 0.0, 1.0)

def is_missing(val):
	return np.isnan(val)

def exclude_missing_data_columns(vector, another_vector):
	new_vector = []
	new_another_vector = []
	for ind in xrange(0,len(vector)):
		if not is_missing(vector[ind]) and not is_missing(another_vector[ind]) :
			new_vector.append(vector[ind])
			new_another_vector.append(another_vector[ind])
	return (np.array(new_vector).astype(float), np.array(new_another_vector).astype(float))

def ordinal_similarity(value, another_value, number_of_values):
	distance = abs(value - another_value)/(number_of_values-1.0)
	return 1.0 - distance

def nominal_similarity(vector, another_vector):
	return np.array([ 1 if x[0] == x[1] else 0 for x in zip(vector, another_vector)])

def try_get(dict, key, default_value):
	return dict[key] if dict.has_key(key) else default_value

def array_to_map(arr):
	return {value: ind for ind,value in enumerate(arr)}

def relabel_ordinal_values_for_feature(records, index, ordered_values):
	ordered_values_map = array_to_map(ordered_values)
	number_of_values = len(ordered_values_map)
	records[:, index] = [try_get(ordered_values_map, x, np.nan) for x in records[:, index]]

def relabel_ordinal_values(records, ordinal_features):
	for feature in ordinal_features:
		relabel_ordinal_values_for_feature(records, feature["index"], feature["ordered_values"])

def relabel_nominal_values(records, nominal_features):
	for feature in nominal_features:
		relabel_ordinal_values_for_feature(records, feature, np.unique(records[:, feature]))

def k_similar_records(records, k, metadata):
	similarity_scores = np.zeros((len(records), len(records)), dtype=[('x', 'float64'), ('y', 'float64')])
	ratio_features = try_get(metadata, "ratio_features", [])
	nominal_features = try_get(metadata, "nominal_features", [])
	ordinal_features = try_get(metadata, "ordinal_features", [])
	euclidean_similarity = lambda x, y: distance_to_similarity(euclidean_distance(x, y))
	similarity_function = cosine_similarity if metadata.has_key("ratio_similarity") and metadata["ratio_similarity"] == "cosine" else euclidean_similarity
	for i in xrange(0, len(records)):
		ith_record_ratio_features = records[i][ratio_features]
		ith_record_nominal_features = records[i][nominal_features]
		for j in xrange(0, len(records)):
			jth_record_ratio_features = records[j][ratio_features]
			jth_record_nominal_features = records[j][nominal_features]
			if i==j:
				similarity_scores[i][j] = (j,0)
			else:
				a_ratio, b_ratio = exclude_missing_data_columns(ith_record_ratio_features, jth_record_ratio_features)
				a_nominal, b_nominal = exclude_missing_data_columns(ith_record_nominal_features, jth_record_nominal_features)
				weighted_ratio_similarity = similarity_function(a_ratio, b_ratio) * len(a_ratio)
				weighted_nominal_similarity = sum(nominal_similarity(a_nominal, b_nominal))
				weighted_ordinal_similarity = 0.0
				valid_ordinal_features = 0
				for feature in ordinal_features:
					val = records[i][feature["index"]]
					another_val = records[j][feature["index"]]
					if not is_missing(val) and not is_missing(another_val):
						weighted_ordinal_similarity += ordinal_similarity(val, another_val, feature["number_of_values"])
						valid_ordinal_features+=1
				number_of_features = float(len(a_ratio) + len(a_nominal) + valid_ordinal_features)
				similarity_scores[i][j] = (j, (weighted_ratio_similarity + weighted_nominal_similarity + weighted_ordinal_similarity)/number_of_features)
	similarity_scores = np.sort(similarity_scores, order='y')[:, ::-1]
	return similarity_scores[:, 0:k]

def log_transform(records, columns):
	for column in columns:
		values = records[:, column]
		# assuming no negative values
		if min(values) == 0:
			values = values + 1.0
		values = np.log(values)
		records[:, column] = values


def print_output(similarity_scores):
	k = len(similarity_scores[0])
	header_line = "Trans Id"
	for i in xrange(1,k+1):
		header_line += ", Id " + str(i) + ", similarity score " + str(i)
	print(header_line)
	for row in xrange(0, len(similarity_scores)):
		row_text = str(row)
		for col in xrange(0, len(similarity_scores[row])):
			record_num = int(similarity_scores[row][col][0])
			score = similarity_scores[row][col][1]
			row_text +=", "+str(record_num)+ ", "+ str(score)
		print(row_text)
			

# Start of execution #

if len(sys.argv) != 4:
	print("Usage: python k_nearest_neighbors.py <k> <Iris|Income> <path_to_data_file.csv>")
	exit(-1)

k = int(sys.argv[1])
dataset_name = sys.argv[2]
file_path = sys.argv[3]

if dataset_name == "Iris":
	columns = [0,1,2,3]
	records = read_csv(file_path, columns).astype(float)
	normalize_data(records, columns)
	print("Similarity table for Iris dataset")
	print_output(k_similar_records(records, metadata = {"ratio_features": columns}, k=k))
else:
	columns_to_read = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
	income_data = read_csv(file_path, columns_to_read)
	ratio_features = [columns_to_read.index(x) for x in [1,3,11,12,13]]
	nominal_features = [columns_to_read.index(x) for x in [2, 6, 7, 8, 9, 10, 14]]
	education_category_feature = {"index": 3, "number_of_values": 16, "ordered_values": list(xrange(1,17))}
	capital_gain_feature = columns_to_read.index(11)
	relabel_nominal_values(income_data, nominal_features)
	relabel_ordinal_values(income_data, [education_category_feature])
	income_data = income_data.astype(float)
	income_metadata = {"ratio_similarity": "cosine","ratio_features": ratio_features, "nominal_features": nominal_features, "ordinal_features": [education_category_feature]}
	log_transform(income_data, [capital_gain_feature])
	normalize_data(income_data, ratio_features)
	print("Similarity table for Income dataset")
	print_output(k_similar_records(income_data, metadata = income_metadata, k=k))