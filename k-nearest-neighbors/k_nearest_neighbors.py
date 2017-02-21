from __future__ import print_function
import csv
import numpy as np
import sys
import math
from scipy.stats import mode
import collections
import warnings
warnings.filterwarnings('ignore')

def read_csv(filename, columns):
	quotes_stripper = lambda x: x.strip('"').strip(" ")
	converter_all_cols = {x:quotes_stripper for x in columns}
	return np.genfromtxt(filename, dtype="string", delimiter=",", skip_header=1, usecols=columns, autostrip=True, converters = converter_all_cols)

def min_max_normalize(column_values, current_min, current_max, new_min, new_max):
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

def normalize_data(records, columns, current_min_max_values = {}):
	for column in columns:
		column_values = records[:, column]
		if not current_min_max_values.has_key(column):
			current_min_max_values[column] = (min(column_values), max(column_values))
		current_min = current_min_max_values[column][0]
		current_max = current_min_max_values[column][1]
		records[:, column] = min_max_normalize(column_values, current_min, current_max, 0.0, 1.0)
	return current_min_max_values

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

def relabel_features(records, index, new_values_map):
	number_of_values = len(new_values_map)
	records[:, index] = [try_get(new_values_map, x, np.nan) for x in records[:, index]]

def relabel_ordinal_values(records, ordinal_features):
	for feature in ordinal_features:
		relabel_features(records, feature["index"], feature["value_mapping"])

def relabel_nominal_values(records, nominal_features_and_values):
	for feature, all_values in nominal_features_and_values:
		relabel_features(records, feature, array_to_map(all_values))

def k_similar_records(test_set, training_set, metadata):
	similarity_scores = np.zeros((len(test_set), len(training_set)), dtype=[('x', 'float64'), ('y', 'float64')])
	ratio_features = try_get(metadata, "ratio_features", [])
	nominal_features = try_get(metadata, "nominal_features", [])
	ordinal_features = try_get(metadata, "ordinal_features", [])
	euclidean_similarity = lambda x, y: distance_to_similarity(euclidean_distance(x, y))
	similarity_function = cosine_similarity if metadata.has_key("ratio_similarity") and metadata["ratio_similarity"] == "cosine" else euclidean_similarity
	for i in xrange(0, len(test_set)):
		ith_record_ratio_features = test_set[i][ratio_features]
		ith_record_nominal_features = test_set[i][nominal_features]
		for j in xrange(0, len(training_set)):
			jth_record_ratio_features = training_set[j][ratio_features]
			jth_record_nominal_features = training_set[j][nominal_features]
			a_ratio, b_ratio = exclude_missing_data_columns(ith_record_ratio_features, jth_record_ratio_features)
			a_nominal, b_nominal = exclude_missing_data_columns(ith_record_nominal_features, jth_record_nominal_features)
			weighted_ratio_similarity = similarity_function(a_ratio, b_ratio) * len(a_ratio)
			weighted_nominal_similarity = sum(nominal_similarity(a_nominal, b_nominal))
			weighted_ordinal_similarity = 0.0
			valid_ordinal_features = 0
			for feature in ordinal_features:
				val = test_set[i][feature["index"]]
				another_val = training_set[j][feature["index"]]
				if not is_missing(val) and not is_missing(another_val):
					weighted_ordinal_similarity += ordinal_similarity(val, another_val, feature["number_of_values"])
					valid_ordinal_features+=1
			number_of_features = float(len(a_ratio) + len(a_nominal) + valid_ordinal_features)
			similarity_scores[i][j] = (j, (weighted_ratio_similarity + weighted_nominal_similarity + weighted_ordinal_similarity)/number_of_features)
	similarity_scores = np.sort(similarity_scores, order='y')[:, ::-1]
	return similarity_scores

def log_transform(records, columns):
	for column in columns:
		values = records[:, column]
		# assuming no negative values
		if min(values) == 0:
			values = values + 1.0
		values = np.log(values)
		records[:, column] = values


def print_similarity_output(similarity_scores):
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

def print_classifier_output(actual, predictions):
	print("Transaction ID, Actual class, Predicted class, Posterior probability")
	for i in xrange(0,len(actual)):
		print(str(i) + ", "+ actual[i]+ ", "+predictions[i][0]+", "+str(predictions[i][1]))

def read_iris_data_set (filename):
	data = read_csv(filename, [0,1,2,3,4])
	records = data[:, 0:4].astype(float)
	classes = data[:, 4]
	return (records, classes)

def map_to_class(similarity_scores, classes):
	return np.vectorize(lambda x: classes[int(x[0])])(similarity_scores)

def knn_classify(k, neighbor_classes):
	k_neighbor_classes = neighbor_classes[:, 0:k]
	mode_results = mode(k_neighbor_classes, axis = 1)
	class_predictions = mode_results.mode[:, 0]
	posterior_probability = mode_results.count[:, 0].astype(float)/float(k)
	return zip(class_predictions, posterior_probability)

def compute_confusion_matrix(actual_classes, predictions, classes, probability = False):
	confusion_matrix = np.zeros((len(classes), len(classes)), dtype = "float")
	predicted_classes = [ x[0] for x in predictions ]
	for i in xrange(0, len(actual_classes)):
		actual_label = actual_classes[i]
		predicted_label = predicted_classes[i]
		confusion_matrix[classes.index(actual_label)][classes.index(predicted_label)] += 1
	if probability:
		actual_classes_count = collections.Counter(actual_classes)
		for i in xrange(0, len(classes)):
			count_of_class = actual_classes_count[classes[i]]
			if count_of_class != 0:
				confusion_matrix[i] = confusion_matrix[i]/float(count_of_class)
	return confusion_matrix

def read_income_data_set(filename):
	pass

# Start of execution #
if len(sys.argv) < 5:
	print("Usage: python k_nearest_neighbors.py <k> <Iris|Income> <path_to_training_file.csv> <path_to_test_file.csv> [should_print_stats]")
	exit(-1)

k = int(sys.argv[1])
dataset_name = sys.argv[2]
training_file_path = sys.argv[3]
test_file_path = sys.argv[4]
should_print_stats = len(sys.argv) > 5
is_iris = dataset_name == "Iris"
if is_iris:
	training_records, training_set_classes = read_iris_data_set(training_file_path)
	columns = [0,1,2,3]
	current_min_max_values = normalize_data(training_records, columns)
	test_records, test_set_classes = read_iris_data_set(test_file_path)
	metadata = {"ratio_features": columns}
	# metadata["ratio_similarity"] = "cosine"
	normalize_data(test_records, columns, current_min_max_values)
else:
	columns_to_read = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	ratio_features = [columns_to_read.index(x) for x in [1,3,11,12,13]]
	nominal_features = [columns_to_read.index(x) for x in [2, 6, 7, 8, 9, 10, 14]]
	education_category_feature = {"index": 3, "number_of_values": 16, "value_mapping": array_to_map(list(xrange(1,17)))}
	class_feature = columns_to_read.index(15)
	capital_gain_feature = columns_to_read.index(11)
	metadata = {"ratio_features": ratio_features, "nominal_features": nominal_features, "ordinal_features": [education_category_feature]}
	# metadata["ratio_similarity"] = "cosine"
	
	training_data = read_csv(training_file_path, columns_to_read)
	training_records = training_data[:, 0:class_feature]
	training_set_classes = training_data[:, class_feature]
	unique_nominal_values = [np.unique(training_records[:, x]) for x in nominal_features]
	relabel_nominal_values(training_records, zip(nominal_features, unique_nominal_values))
	relabel_ordinal_values(training_records, [education_category_feature])
	training_records = training_records.astype(float)
	log_transform(training_records, [capital_gain_feature])
	current_min_max_values = normalize_data(training_records, ratio_features)

	test_data = read_csv(test_file_path, columns_to_read)
	test_records = test_data[:, 0:class_feature]
	test_set_classes = test_data[:, class_feature]
	for ind, feature in enumerate(nominal_features):
		new_values_in_test_data = [ x for x in test_data[:, feature] if x not in unique_nominal_values[ind]]
		if len(new_values_in_test_data) > 0:
			unique_nominal_values[ind] = np.concatenate((unique_nominal_values[ind], new_values_in_test_data))
	relabel_nominal_values(test_records, zip(nominal_features, unique_nominal_values))
	relabel_ordinal_values(test_records, [education_category_feature])
	test_records = test_records.astype(float)
	log_transform(test_records, [capital_gain_feature])
	normalize_data(test_records, ratio_features, current_min_max_values)

similarity_scores = k_similar_records(test_records, training_records, metadata = metadata)
neighbor_classes = map_to_class(similarity_scores, training_set_classes)
predictions = knn_classify(k, neighbor_classes)
# print_classifier_output(test_set_classes, predictions)
if should_print_stats:
	if is_iris:
		confusion = compute_confusion_matrix(test_set_classes, predictions, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"], probability = True)
		print("\n\nConfusion matrix")
		print("Iris-setosa " + "Iris-versicolor " +  "Iris-virginica")
		print(confusion)
		accuracy = (confusion[0][0]+confusion[1][1] + confusion[2][2])/sum(sum(confusion))
		print("k="+str(k))
		print("Accuracy: " + str(accuracy))
	else:
		confusion = compute_confusion_matrix(test_set_classes, predictions, [">50K", "<=50K"]).astype(float)
		print("\n\nConfusion matrix")
		print(">50K" + " <=50K")
		print(confusion)
		accuracy = (confusion[0][0]+confusion[1][1])/sum(sum(confusion))
		precision = confusion[0][0]/sum(confusion[:, 0])
		recall = confusion[0][0]/sum(confusion[0, :])
		f_score = 2*precision*recall/(precision+recall)
		print("k="+str(k))
		print("Accuracy: "+ str(accuracy))
		print("Precision: "+ str(precision))
		print("Recall: "+ str(recall))
		print("F: "+ str(f_score))