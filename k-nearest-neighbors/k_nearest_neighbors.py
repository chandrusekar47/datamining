from __future__ import print_function
import csv
import numpy as np
import sys
import math
from scipy.stats import mode
import collections
import warnings
from util import *
import iris
import income
warnings.filterwarnings('ignore')

# Start of execution #
if len(sys.argv) < 5:
	print("Usage: python k_nearest_neighbors.py <k> <Iris|Income> <path_to_training_file.csv> <path_to_test_file.csv> [extended_run]")
	exit(-1)

k = int(sys.argv[1])
dataset_name = sys.argv[2]
training_file_path = sys.argv[3]
test_file_path = sys.argv[4]
extended_run = len(sys.argv) > 5
dataset = iris if dataset_name == "Iris" else income
(training_records, training_set_classes, test_records, test_set_classes) = dataset.prepare_training_and_test(training_file_path, test_file_path)
similarity_scores = k_similar_records(test_records, training_records, metadata = dataset.metadata())
neighbor_classes = map_to_class(similarity_scores, training_set_classes)

if extended_run:
	k_values = xrange(5,25)
	results = repeat_knn(neighbor_classes, k_values, test_set_classes, dataset.class_values())
	dataset.print_csv(results, test_set_classes)
	# for k in np.sort(results.keys()):
	# 	dataset.print_stats(k, results[k])
else:
	predictions = knn_classify(k, neighbor_classes)
	predicted_classes = [ x[0] for x in predictions]
	print_classifier_output(test_set_classes, predictions)