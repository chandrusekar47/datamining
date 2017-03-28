import csv
import numpy as np
import sys
import math
from scipy.stats import mode
from collections import namedtuple
from collections import Counter

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

def log_transform(records, columns):
	for column in columns:
		values = records[:, column]
		# assuming no negative values
		if min(values) == 0:
			values = values + 1.0
		values = np.log(values)
		records[:, column] = values

def map_to_class(similarity_scores, classes):
	return np.vectorize(lambda x: classes[int(x[0])])(similarity_scores)

def compute_confusion_matrix(actual_classes, predicted_classes, classes, probability = False):
	confusion_matrix = np.zeros((len(classes), len(classes)), dtype = "float")
	for i in xrange(0, len(actual_classes)):
		actual_label = actual_classes[i]
		predicted_label = predicted_classes[i]
		confusion_matrix[classes.index(actual_label)][classes.index(predicted_label)] += 1
	if probability:
		actual_classes_count = Counter(actual_classes)
		for i in xrange(0, len(classes)):
			count_of_class = actual_classes_count[classes[i]]
			if count_of_class != 0:
				confusion_matrix[i] = confusion_matrix[i]/float(count_of_class)
	return confusion_matrix

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

def unit_normalize_vector(vector):
	return vector/magnitude(vector)

def k_similar_records(test_set, training_set, metadata):
	similarity_scores = np.zeros((len(test_set), len(training_set)), dtype=[('x', 'float64'), ('y', 'float64')])
	ratio_features = try_get(metadata, "ratio_features", [])
	nominal_features = try_get(metadata, "nominal_features", [])
	ordinal_features = try_get(metadata, "ordinal_features", [])
	if metadata.has_key("ratio_similarity") and metadata["ratio_similarity"] == "cosine":
		similarity_function = cosine_similarity
	else:
		r = int(metadata["minkowski_r"])
		if metadata["unit_normalize"]:
			similarity_function = lambda x, y: distance_to_similarity(minkowski_distance(unit_normalize_vector(x), unit_normalize_vector(y), r))
		else:
			similarity_function = lambda x, y: distance_to_similarity(minkowski_distance(x, y, r))
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

def knn_classify(k, neighbor_classes):
	k_neighbor_classes = neighbor_classes[:, 0:k]
	mode_results = mode(k_neighbor_classes, axis = 1)
	class_predictions = mode_results.mode[:, 0]
	posterior_probability = mode_results.count[:, 0].astype(float)/float(k)
	return zip(class_predictions, posterior_probability)

general_stats = namedtuple('general_stats', ['confusion_matrix', 'accuracy', 'error'])
binary_stats = namedtuple('binary_stats', ['confusion_matrix', 'accuracy', 'error','tpr', 'fpr', 'fnr', 'tnr', 'recall', 'precision', 'f_score', 'confidence_scores'])

def get_general_stats(actual_classes, predicted_classes, class_values, probabilities = []):
	confusion_matrix = compute_confusion_matrix(actual_classes, predicted_classes, class_values, probability = True)
	accuracy = sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
	return general_stats(confusion_matrix = confusion_matrix,
							accuracy = accuracy,
							error = 1.0-accuracy) 

def generate_confidence_scores(predicted_classes, probabilities):
	positive_class = ">50K"
	return map(lambda x: x[1] if x[0]=='>50K' else 1-x[1], zip(predicted_classes, probabilities))

def get_binary_stats(actual_classes, predicted_classes, class_values, probabilities):
	stats = get_general_stats(actual_classes, predicted_classes, class_values)
	recall = tpr = stats.confusion_matrix[0][0]
	fpr = stats.confusion_matrix[1][0]
	precision = stats.confusion_matrix[0][0]/sum(stats.confusion_matrix[:, 0])
	f_score = 2*precision*recall/(precision + recall)
	return binary_stats(confusion_matrix = stats.confusion_matrix,
							accuracy = stats.accuracy,
							error = stats.error,
							recall = recall,
							precision = precision,
							f_score = f_score,
							tpr = tpr,
							fpr = fpr,
							tnr = 1.0 - fpr,
							fnr = 1.0 - tpr,
							confidence_scores = generate_confidence_scores(predicted_classes, probabilities))

def repeat_knn(all_neighbor_classes, k_values, actual_classes, class_values):
	stats_for_all_k = {}
	stats_method = get_binary_stats if len(class_values) == 2 else get_general_stats
	for k in k_values:
		predictions = knn_classify(k, all_neighbor_classes)
		predicted_classes = [ x[0] for x in predictions]
		probabilities = [ x[1] for x in predictions]
		stats_for_all_k[k] = stats_method(actual_classes, predicted_classes, class_values, probabilities = probabilities)
	return stats_for_all_k

# def threshold_and_generate_all_rates():
# 	pass

