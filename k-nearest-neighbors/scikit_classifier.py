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
from sklearn.neighbors import KNeighborsClassifier
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

dataset = income
(training_records, training_set_classes, test_records, test_set_classes) = dataset.prepare_training_and_test(training_file_path, test_file_path)
stats_map = {}

for k in xrange(5,25):
	classifier = KNeighborsClassifier(n_neighbors = k, weights='uniform', p=2)
	classifier.fit(training_records, training_set_classes)

	predicted_classes = classifier.predict(test_records)
	prediction_probabs = np.max(classifier.predict_proba(test_records), axis=1)
	stats_map[k] = get_binary_stats(test_set_classes, predicted_classes, dataset.class_values(), prediction_probabs)

dataset.print_csv(stats_map, test_set_classes)
