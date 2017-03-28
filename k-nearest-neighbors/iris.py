from util import *

IRIS__COLUMNS = {
	"sepal_length": 0,
	"sepal_width": 1,
	"petal_length": 2,
	"petal_width": 3,
	"class": 4
}

def read_iris_data_set (filename):
	data = read_csv(filename, [0,1,2,3,4])
	records = data[:, 0:4].astype(float)
	classes = data[:, 4]
	return (records, classes)

def column(col_name):
	return IRIS__COLUMNS[col_name]

def metadata():
	meta = {"ratio_features": map(column, ["sepal_length", "sepal_width", "petal_length", "petal_width"])}
	# meta["ratio_similarity"] = "cosine"
	meta["ratio_simialarity"] = "minkowski"
	meta["minkowski_r"] = 2
	meta["unit_normalize"] = False
	return meta

def prepare_training_and_test(training_file_path, test_file_path):
	training_records, training_set_classes = read_iris_data_set(training_file_path)
	columns = metadata()["ratio_features"]
	current_min_max_values = normalize_data(training_records, columns)
	test_records, test_set_classes = read_iris_data_set(test_file_path)
	normalize_data(test_records, columns, current_min_max_values)
	return (training_records, training_set_classes, test_records, test_set_classes)

def print_stats(k, stats):
	print("\n\nConfusion matrix")
	print("Iris-setosa " + "Iris-versicolor " +  "Iris-virginica")
	print(np.round(stats.confusion_matrix, 2))
	print("k="+str(k))
	print("Accuracy: " + str(np.round(stats.accuracy,2)))
	print("blah");
	print(",".join([str(x) for x in np.diag(stats.confusion_matrix)]));

def class_values():
	return ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

def print_csv(results, actual_classes = []):
	print("k, accuracy, error")
	for k in np.sort(results.keys()):
		stats = results[k]
		values = [str(np.round(x, 2)) for x in [k, stats.accuracy, stats.error]]
		print(",".join(values))
