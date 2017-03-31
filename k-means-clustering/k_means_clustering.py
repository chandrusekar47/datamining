import numpy as np
import os 
import sys
import scipy.stats
from collections import Counter
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import scipy.stats as stats
import time
import scipy.cluster.vq as vq
from sklearn.cluster import Birch

colors = ["blue" , "green" , "red" , "orange" , "black" ]
markers = ["o", "v", "D", "s", "p", "P", "*"]

def compute_distance(points, centroid):
	return np.sqrt(np.sum((points - centroid)**2, axis=1))

def compute_clusters_and_centroid (points, centroids):
	distance_to_all_centroids = np.zeros((points.shape[0], centroids.shape[0]))
	for index, centroid in enumerate(centroids):
		distance_to_all_centroids[:, index] = compute_distance(points, centroid)
	new_centroids = np.zeros(centroids.shape)
	cluster_numbers = np.argmin(distance_to_all_centroids, axis = 1)
	non_empty_clusters = []
	for index in range(0, centroids.shape[0]):
		points_in_cluster = points[cluster_numbers == index, :]
		if len(points_in_cluster) == 0:
			print("Found empty cluster")
			continue
		non_empty_clusters.append(index)
		new_centroids[index] = np.mean(points_in_cluster, axis = 0)
	new_centroids = new_centroids[non_empty_clusters, :]
	return (cluster_numbers, new_centroids)

def compute_sse(points, cluster_numbers, at_point_level = False):
	unique_clusters = np.unique(cluster_numbers)
	unique_clusters.sort()
	sse = np.zeros(len(unique_clusters))
	sse_by_points = np.zeros(len(points))
	for ind, cluster in enumerate(unique_clusters):
		points_in_cluster = points[cluster_numbers == cluster, :]
		center_of_cluster = np.mean(points_in_cluster, axis = 0)
		sse_by_points[cluster_numbers == cluster] = compute_distance(points_in_cluster, center_of_cluster) ** 2
		sse[ind] = (sse_by_points[cluster_numbers == cluster]).sum()
	if at_point_level:
		return (sse, sse.sum(), sse_by_points)
	else:
		return (sse, sse.sum())

def compute_ssb(points, cluster_numbers):
	unique_clusters = np.unique(cluster_numbers)
	unique_clusters.sort()
	ssb = np.zeros(len(unique_clusters))
	global_mean = np.mean(points, axis = 0)
	for ind, cluster in enumerate(unique_clusters):
		points_in_cluster = points[cluster_numbers == cluster, :]
		center_of_cluster = np.mean(points_in_cluster, axis = 0)
		ssb[ind] = points_in_cluster.shape[0] * (compute_distance([center_of_cluster], global_mean).sum())**2
	return (ssb, ssb.sum())

def compute_all_pair_distances(points):
	row_duplicated_matrix = np.repeat([points], points.shape[0], axis = 1).reshape(points.shape[0],points.shape[0],points.shape[1])
	col_duplicated_matrix = np.repeat([points], points.shape[0], axis = 0)
	sq_diff_between_points = (row_duplicated_matrix - col_duplicated_matrix)**2
	sum_of_square_distances = np.sqrt(sq_diff_between_points.sum(axis=2))
	return sum_of_square_distances

def silhouette_width(points, cluster_numbers, all_pair_distances):
	unique_clusters = np.unique(cluster_numbers)
	unique_clusters.sort()
	intra_cluster_avg_distance = np.zeros(points.shape[0])
	inter_cluster_avg_distance = np.ones(points.shape[0]) * (10**9)
	point_sw = np.zeros(points.shape[0])
	cluster_sw = np.zeros(len(unique_clusters))
	for ind, cluster in enumerate(unique_clusters):
		points_in_cluster = cluster_numbers == cluster
		indices_of_c = np.argwhere(cluster_numbers == cluster).flatten()
		intra_cluster_avg_distance[indices_of_c] = all_pair_distances[indices_of_c, :][:, indices_of_c].sum(axis = 0)/(len(indices_of_c)-1)
		for other_cluster in unique_clusters:
			if cluster == other_cluster:
				continue
			points_in_other_cluster = cluster_numbers == other_cluster
			indices_of_o_c = np.argwhere(cluster_numbers == other_cluster).flatten()
			dist_to_o_c = all_pair_distances[indices_of_o_c, :][:, indices_of_c].mean(axis = 0)
			inter_cluster_avg_distance[indices_of_c] = np.minimum(inter_cluster_avg_distance[indices_of_c], dist_to_o_c)
		point_sw[indices_of_c] = (inter_cluster_avg_distance[indices_of_c] - intra_cluster_avg_distance[indices_of_c])/np.maximum(inter_cluster_avg_distance[indices_of_c], intra_cluster_avg_distance[indices_of_c])
		cluster_sw[ind] = point_sw[indices_of_c].mean()
	return (cluster_sw, point_sw.mean())

def calc_confusion_matrix(actual_class, predicted_class):
	unique_classes = np.unique(actual_class).tolist()
	unique_classes.sort()
	confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)))
	for ind, klazz in enumerate(unique_classes):
		counts = Counter(predicted_class[actual_class == klazz])
		for klazz, count in counts.iteritems():
			confusion_matrix[ind][unique_classes.index(klazz)] += count
	confusion_matrix = confusion_matrix / confusion_matrix.sum(axis = 1).reshape(len(unique_classes), 1)
	return confusion_matrix

def min_max_normalize(column_values, current_min, current_max, new_min, new_max):
	old_range = current_max - current_min
	new_range = new_max - new_min
	normalize_func = lambda x: new_min + (x-current_min)*new_range/(old_range) if not np.isnan(x) else np.nan
	return np.vectorize(normalize_func)(column_values)

def normalize_data(records, columns):
	for column in columns:
		column_values = records[:, column]
		(current_min, current_max) = (min(column_values), max(column_values))
		records[:, column] = min_max_normalize(column_values, current_min, current_max, 0.0, 1.0)

def plot_values(title, xlab, ylab, points, cluster_numbers, true_cluster_numbers, predicted_class, filename):
	true_classes = np.unique(true_cluster_numbers).tolist()
	class_colors = colors[0:len(true_classes)]
	predicted_clusters = np.unique(cluster_numbers)
	cluster_markers = markers[0:len(predicted_clusters)]
	fig, axis = plt.subplots()
	axis.set_title(title)
	axis.set_xlabel(xlab)
	axis.set_ylabel(ylab)
	for ind, klazz in enumerate(true_classes):
		true_class_points = points[true_cluster_numbers == klazz, :]
		cluster_hull = ConvexHull(true_class_points)
		for simplex in cluster_hull.simplices:
			axis.plot(true_class_points[simplex, 0], true_class_points[simplex, 1], linestyle = ':', c = class_colors[ind])

	for ind, cluster in enumerate(predicted_clusters):
		predicted_cluster_points = points[cluster_numbers == cluster, :]
		cluster_hull = ConvexHull(predicted_cluster_points)
		klazz = np.unique(predicted_class[cluster_numbers == cluster])[0]
		for simplex in cluster_hull.simplices:
			axis.plot(predicted_cluster_points[simplex, 0], predicted_cluster_points[simplex, 1], linestyle = 'solid', c = class_colors[true_classes.index(klazz)])
		misclassified_points = predicted_cluster_points[predicted_class[cluster_numbers == cluster] != true_cluster_numbers[cluster_numbers == cluster]]
		axis.plot(misclassified_points[:, 0], misclassified_points[:, 1], c="black", markerfacecolor='None', marker='o', linestyle = 'None', markersize = 25.0)
		axis.plot(predicted_cluster_points[:, 0], predicted_cluster_points[:, 1], marker = cluster_markers[ind], linestyle = 'None', c = class_colors[true_classes.index(klazz)])
	axis.hold(False)
	mng = plt.get_current_fig_manager()
	size = mng.window.maxsize()
	fig.show()
	mng.resize(width = size[0]*0.5, height = size[1]*0.75)
	fig.canvas.draw()
	fig.savefig(filename)

def entropy(numbers):
	counts = Counter(numbers)
	return stats.entropy(map(lambda x: x[1]/float(len(numbers)), Counter(numbers).iteritems()), base = 2)

def gini(numbers):
	counts = Counter(numbers)
	return 1 - np.sum(map(lambda x: (x[1]/float(len(numbers)))**2, Counter(numbers).iteritems()))

def compute_quality_metrics(cluster_numbers, quality_attribute):
	unique_clusters = np.unique(cluster_numbers).tolist()
	unique_clusters.sort()
	quality_numbers = []
	print("cluster, mean, variance, entropy, gini, min, max")
	for ind, cluster in enumerate(unique_clusters):
		quality_of_cluster = quality_attribute[cluster_numbers == cluster]
		print("Cluster " + str(cluster))
		for quality_val, count in Counter(quality_of_cluster).iteritems():
			print("Quality :%s, Count: %s"%(quality_val, count))
		quality_numbers.append({
				"cluster": cluster,
				"mean": np.mean(quality_of_cluster),
				"variance": np.var(quality_of_cluster),
				"entropy": entropy(quality_of_cluster),
				"gini": gini(quality_of_cluster),
				"min": np.min(quality_of_cluster),
				"max": np.max(quality_of_cluster)
			})
		print("%s, %0.3f, %0.3f, %0.3f, %0.3f, %s, %s"%(quality_numbers[ind]["cluster"], quality_numbers[ind]["mean"], quality_numbers[ind]["variance"], quality_numbers[ind]["entropy"], quality_numbers[ind]["gini"], quality_numbers[ind]["min"], quality_numbers[ind]["max"]))
	return quality_numbers

def generate_new_centroids(points, cluster_numbers, num_of_centroids):
	(_, _, sse_by_points) = compute_sse(points, cluster_numbers, at_point_level = True)
	return points[sse_by_points.argsort()[-num_of_centroids:][::-1], :]


def k_means_clustering(points, centroids):
	k = centroids.shape[0]
	while True:
		(cluster_numbers, new_centroids) = compute_clusters_and_centroid (points, centroids)
		if new_centroids.shape[0] < k:
			generated_centroids = generate_new_centroids(points, cluster_numbers, k - new_centroids.shape[0])
			new_centroids = np.append(new_centroids, generated_centroids, axis = 0)
		elif (new_centroids == centroids).all():
			break
		centroids = new_centroids
	return (cluster_numbers, centroids)

def assign_classes(cluster_numbers, true_cluster_numbers):
	unique_classes = np.unique(cluster_numbers)
	classes = np.zeros(len(cluster_numbers))
	for cluster in cluster_numbers:
		points_in_cluster = cluster_numbers == cluster
		majority_class = scipy.stats.mode(true_cluster_numbers[points_in_cluster]).mode[0]
		classes[points_in_cluster] = majority_class
	return classes

def read_csv(filename, converters = None):
	return np.loadtxt(filename, delimiter=',', skiprows = 1, converters = converters)

def choose_k_centroids(points, k):
	return points[np.random.randint(low=0, high = points.shape[0], size = k), :]

def file_name(str):
	return "".join([c for c in str if c.isalpha() or c.isdigit() or c=='']).rstrip()

def main(filename, converters, point_columns, true_cluster_column, id_column, should_normalize = False, k = 2):
	data = read_csv(filename, converters = converters)
	points = data[:, point_columns]
	true_cluster_numbers = data[:, true_cluster_column]
	record_ids = data[:, 0]
	if should_normalize:
		normalize_data(points, range(0, len(point_columns)))
	centroids = choose_k_centroids(points, k)
	(cluster_numbers, new_centroids) = k_means_clustering(points, centroids)
	# (new_centroids, cluster_numbers) = vq.kmeans2(points, k=k, minit = 'random')
	assigned_classes = assign_classes(cluster_numbers, true_cluster_numbers)
	all_pair_distances = compute_all_pair_distances(points)
	# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
	# ret_val = (compute_sse(points, cluster_numbers)[1], compute_ssb(points, cluster_numbers)[1], silhouette_width(points, cluster_numbers, all_pair_distances)[1], calc_confusion_matrix(true_cluster_numbers, assigned_classes)[0][0], calc_confusion_matrix(true_cluster_numbers, assigned_classes)[1][1])
	# print("k="+str(k))
	# print("True cluster level SSE: %s, True overall cluster SSE: %0.4f"%(compute_sse(points, true_cluster_numbers)))
	# print("True cluster level SSB: %s, True overall cluster SSB: %0.4f"%(compute_ssb(points, true_cluster_numbers)))
	# print("Silhouette width by cluster: %s, Silhouette width: %0.4f"%(silhouette_width(points, true_cluster_numbers, all_pair_distances)))
	# print("Predicted cluster level SSE: %s, Predicted overall cluster SSE: %0.4f"%(compute_sse(points, cluster_numbers)))
	# print("Predicted cluster level SSB: %s, Predicted overall cluster SSB: %0.4f"%(compute_ssb(points, cluster_numbers)))
	# print("Silhouette width by cluster: %s, Silhouette width: %0.4f"%(silhouette_width(points, cluster_numbers, all_pair_distances)))
	# print("Confusion matrix: %s"%(calc_confusion_matrix(true_cluster_numbers, assigned_classes)))
	# if len(point_columns) == 2:
	# 	fig_filename = os.path.join('images', "k_%s_%s_%s.png"%(k, file_name(filename), int(time.time())))
	# 	plot_values('Scatter plot between features X1 and X2','X1', 'X2', points, cluster_numbers, true_cluster_numbers, assigned_classes, fig_filename)
	# else:
	# 	compute_quality_metrics(cluster_numbers, data[:, 12])
	print("row ID, cluster number, predicted true cluster number")
	for ind,point in enumerate(points):
		print("%s, %s, %s"%(int(record_ids[ind]), cluster_numbers[ind], int(assigned_classes[ind])))
	# return ret_val

if __name__ == '__main__':
	id_converter = lambda x: float(x.strip('"'))
	datasets = {
		'Easy': ['data/TwoDimEasy.csv', {0: id_converter}, [1,2], 3, 0],
		'Hard': ['data/TwoDimHard.csv', {0: id_converter}, [1,2], 3, 0],
		'Wine': ['data/wine_quality-red.csv', {0: id_converter, 13: lambda x: 0 if x.strip('"') == "Low" else 1}, range(1, 12), 13, 0, True],
	}
	if len(sys.argv) != 3 or (not datasets.has_key(sys.argv[1])):
		print("Usage: python k_means_clustering.py <Easy|Hard|Wine> <k-value>")
		exit()
	dataset_chosen = sys.argv[1]
	k = int(sys.argv[2])
	if dataset_chosen == "Wine":
		# k_values = [17,18,19,20,21,22]
		# output_of_runs = np.zeros((len(k_values), 5))
		# for i in xrange(0,30):
		# 	for ind, k in enumerate(k_values):
		# 		ret_val = main(*datasets[dataset_chosen], k=k)
		# 		ret_val = np.array([x for x in ret_val])
		# 		output_of_runs[ind, :] += ret_val
		# output_of_runs = output_of_runs / 30.0
		# print("k, avg sse, avg ssb, s_w, tpr, tnr")
		# for ind,k in enumerate(k_values):
		# 	print("%s, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f"%(k, output_of_runs[ind][0], output_of_runs[ind][1], output_of_runs[ind][2], output_of_runs[ind][3], output_of_runs[ind][4]))
		main(*datasets[dataset_chosen], k=k)
	else:
		main(*datasets[dataset_chosen], k=k)