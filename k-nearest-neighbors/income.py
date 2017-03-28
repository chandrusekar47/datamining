from util import *
from scipy.stats import mode
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

INCOME_FILE_COLUMNS = {
	"ID": 0,
	"age": 1,
	"workclass": 2,
	"fnlwgt": 3,
	"education": 4,
	"education_cat": 5,
	"marital_status": 6,
	"occupation": 7,
	"relationship": 8,
	"race": 9,
	"gender": 10,
	"capital_gain": 11,
	"capital_loss": 12,
	"hour_per_week": 13,
	"native_country": 14,
	"class": 15
}
INCOME_DATASET_COLUMNS = {
	"age": 0,
	"workclass": 1,
	"fnlwgt": 2,
	"education_cat": 3,
	"marital_status": 4,
	"occupation": 5,
	"relationship": 6,
	"race": 7,
	"gender": 8,
	"capital_gain": 9,
	"capital_loss": 10,
	"hour_per_week": 11,
	"native_country": 12,
	"class": 13
}

def file_column(col_name):
	return INCOME_FILE_COLUMNS[col_name]

def column(col_name):
	return INCOME_DATASET_COLUMNS[col_name]

def metadata():
	ratio_features = ["age", "fnlwgt", "capital_gain", "capital_loss", "hour_per_week"]
	nominal_features = ["workclass", "marital_status", "occupation", "relationship", "race", "gender", "native_country"]
	education_category_feature = {"index": column("education_cat"), "number_of_values": 16, "value_mapping": array_to_map([str(x) for x in xrange(1,17)])}
	meta = {
		"ratio_features": map(column, ratio_features), 
		"nominal_features": map(column, nominal_features), 
		"ordinal_features": [education_category_feature]
	}
	# meta["ratio_similarity"] = "cosine"
	# eculidean
	meta["ratio_simialarity"] = "minkowski"
	meta["minkowski_r"] = 2
	meta["unit_normalize"] = True
	return meta

def impute_missing_values(data, categorical_features):
	# return;
	for feature in categorical_features:
		most_frequent_value = mode(data[:, feature]).mode[0]
		data[data[:, feature] == '?', feature] = most_frequent_value

def prepare_training_and_test(training_file_path, test_file_path):
	columns_to_read = np.sort(map(file_column, INCOME_DATASET_COLUMNS.keys()))
	training_data = read_csv(training_file_path, columns_to_read)
	training_records = training_data[:, 0:column("class")]
	training_set_classes = training_data[:, column("class")]
	m = metadata()
	impute_missing_values(training_records, m["nominal_features"])
	unique_nominal_values = [np.unique(training_records[:, x]) for x in m["nominal_features"]]
	relabel_nominal_values(training_records, zip(m["nominal_features"], unique_nominal_values))
	relabel_ordinal_values(training_records, m["ordinal_features"])
	training_records = training_records.astype(float)
	log_transform(training_records, [column("capital_gain")])
	current_min_max_values = normalize_data(training_records, m["ratio_features"])

	test_data = read_csv(test_file_path, columns_to_read)
	test_records = test_data[:, 0:column("class")]
	test_set_classes = test_data[:, column("class")]
	impute_missing_values(test_records, m["nominal_features"])
	for ind, feature in enumerate(m["nominal_features"]):
		new_values_in_test_data = [ x for x in test_data[:, feature] if x not in unique_nominal_values[ind]]
		if len(new_values_in_test_data) > 0:
			unique_nominal_values[ind] = np.concatenate((unique_nominal_values[ind], new_values_in_test_data))
	relabel_nominal_values(test_records, zip(m["nominal_features"], unique_nominal_values))
	relabel_ordinal_values(test_records, m["ordinal_features"])
	test_records = test_records.astype(float)
	log_transform(test_records, [column("capital_gain")])
	normalize_data(test_records, m["ratio_features"], current_min_max_values)
	return (training_records, training_set_classes, test_records, test_set_classes)

def print_stats(k, stats):
	print("\n\nConfusion matrix")
	print(">50K" + " <=50K")
	print(np.round(stats.confusion_matrix,2))
	print("k="+str(k))
	print("Accuracy: "+ str(np.round(stats.accuracy,2)))
	print("Precision: "+ str(np.round(stats.precision,2)))
	print("Recall: "+ str(np.round(stats.recall,2)))
	print("F: "+ str(np.round(stats.f_score,2)))
	print("TPR: " + str(np.round(stats.tpr,2)))
	print("FPR: " + str(np.round(stats.fpr,2)))
	print("FNR: " + str(np.round(stats.fnr,2)))
	print("TNR: " + str(np.round(stats.tnr,2)))

def print_csv(results, actual_classes):
	print("k, accuracy, precision, recall, f_score, tpr, fpr, fnr, tnr")
	for k in np.sort(results.keys()):
		stats = results[k]
		values = [str(np.round(x, 2)) for x in [k, stats.accuracy, stats.precision, stats.recall, stats.f_score, stats.tpr, stats.fpr, stats.fnr, stats.tnr]]
		print(",".join(values))
		generate_roc_curve(actual_classes, stats.confidence_scores, 'roc_curve_cosine_'+str(k)+'.png', k)

def class_values():
	return [">50K", "<=50K"]

def generate_roc_curve(actual_classes, confidence_scores, filename, k):
	fpr, tpr, thresholds = roc_curve(map(lambda x:1-class_values().index(x), actual_classes), confidence_scores, drop_intermediate = False)
	fig, axis = plt.subplots(figsize=(3,3))
	axis.set_title("ROC curve for k = "+str(k))
	axis.set_xlabel('FPR')
	axis.set_ylabel('TPR')
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	fig.tight_layout()
	plt.savefig('./images/' + filename)
	# plt.show()

