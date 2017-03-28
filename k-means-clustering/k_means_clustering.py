import numpy as np
import sys
import scipy.stats
from collections import Counter
from matplotlib import pyplot as plt

def compute_distance(points, centroid):
	return np.sqrt(np.sum((points - centroid)**2, axis=1))

def compute_clusters_and_centroid (points, centroids):
	distance_to_all_centroids = np.zeros((points.shape[0], centroids.shape[0]))
	for index, centroid in enumerate(centroids):
		distance_to_all_centroids[:, index] = compute_distance(points, centroid)
	new_centroids = np.zeros(centroids.shape)
	cluster_numbers = np.argmin(distance_to_all_centroids, axis = 1)
	for index in range(0, centroids.shape[0]):
		points_in_cluster = points[cluster_numbers == index, :]
		new_centroids[index] = np.mean(points_in_cluster, axis = 0)
	return (cluster_numbers, new_centroids)

def compute_sse(points, cluster_numbers):
	unique_clusters = np.unique(cluster_numbers)
	sse = np.zeros(len(unique_clusters))
	for ind, cluster in enumerate(unique_clusters):
		points_in_cluster = points[cluster_numbers == cluster, :]
		center_of_cluster = np.mean(points_in_cluster, axis = 0)
		sse[ind] = (compute_distance(points_in_cluster, center_of_cluster) ** 2).sum()
	return (sse, sse.sum())

def compute_ssb(points, cluster_numbers):
	unique_clusters = np.unique(cluster_numbers)
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

def plot_values(title, xlab, ylab, points, cluster_numbers, true_cluster_numbers):
	fig, axis = plt.subplots()
	axis.set_title(title)
	axis.set_xlabel(xlab)
	axis.set_ylabel(ylab)
	colors = np.chararray(points.shape[0])
	colors[:] = 'blue'
	colors[true_cluster_numbers == 2] = 'red'
	points_in_cluster_1 = cluster_numbers == 0
	points_in_cluster_2 = cluster_numbers == 1
	axis.hold(True)
	axis.plot(points[true_cluster_numbers == 1, 0], points[true_cluster_numbers == 1, 1], marker = 'o', markeredgecolor = "red", markerfacecolor = 'None', markersize = 35.0, linestyle = 'None')
	axis.plot(points[true_cluster_numbers == 2, 0], points[true_cluster_numbers == 2, 1], marker = 'o', markeredgecolor = "blue", markerfacecolor = 'None', markersize = 35.0, linestyle = 'None')
	axis.plot(points[points_in_cluster_1, 0], points[points_in_cluster_1, 1], marker = "+", linestyle='None', c='black', markersize=10.0, markerfacecolor='None')
	axis.plot(points[points_in_cluster_2, 0], points[points_in_cluster_2, 1], marker = "D", linestyle='None', c='black', markersize=10.0)
	axis.hold(False)
	fig.show()
	plt.show()

def k_means_clustering(points, centroids):
	while True:
		(cluster_numbers, new_centroids) = compute_clusters_and_centroid (points, centroids)
		if (new_centroids == centroids).all():
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

def main(filename, converters, point_columns, true_cluster_column, id_column, k):
	data = read_csv(filename, converters = converters)
	points = data[:, point_columns]
	true_cluster_numbers = data[:, true_cluster_column]
	centroids = choose_k_centroids(points, k)
	(cluster_numbers, new_centroids) = k_means_clustering(points, centroids)
	assigned_classes = assign_classes(cluster_numbers, true_cluster_numbers)
	all_pair_distances = compute_all_pair_distances(points)
	print("True cluster level SSE: %s, True overall cluster SSE: %s"%(compute_sse(points, true_cluster_numbers)))
	print("True cluster level SSB: %s, True overall cluster SSB: %s"%(compute_ssb(points, true_cluster_numbers)))
	print("Silhouette width by cluster: %s, Silhouette width: %s"%(silhouette_width(points, true_cluster_numbers, all_pair_distances)))
	print("Predicted cluster level SSE: %s, Predicted overall cluster SSE: %s"%(compute_sse(points, cluster_numbers)))
	print("Predicted cluster level SSB: %s, Predicted overall cluster SSB: %s"%(compute_ssb(points, cluster_numbers)))
	print("Silhouette width by cluster: %s, Silhouette width: %s"%(silhouette_width(points, cluster_numbers, all_pair_distances)))
	print("Confusion matrix: %s"%(calc_confusion_matrix(true_cluster_numbers, assigned_classes)))
	if len(point_columns) == 2:
		plot_values('Scatter plot between features X1 and X2','X1', 'X2', points, cluster_numbers, true_cluster_numbers)
	return (points, assigned_classes, cluster_numbers, true_cluster_numbers)

if __name__ == '__main__':
	id_converter = lambda x: float(x.strip('"'))
	datasets = {
		'Easy': ['data/TwoDimEasy.csv', {0: id_converter}, [1,2], 3, 0],
		'Hard': ['data/TwoDimHard.csv', {0: id_converter}, [1,2], 3, 0],
		'Wine': ['data/wine_quality-red.csv', {0: id_converter, 13: lambda x: 0 if x.strip('"') == "Low" else 1}, range(1, 12), 13, 0],
	}
	if len(sys.argv) != 3 or (not datasets.has_key(sys.argv[1])):
		print("Usage: python k_means_clustering.py <Easy|Hard|Wine> <k-value>")
		exit()
	dataset_chosen = sys.argv[1]
	k = int(sys.argv[2])
	main(*datasets[dataset_chosen], k=k)