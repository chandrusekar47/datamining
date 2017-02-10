To run
---------
Please copy Iris.csv and Income.csv to the folder containing k_nearest_neighbors.py. Use the following command to run the program

bash run.sh

To change k
---------
open run.sh, change the value 5 to whatever the new value of k should be


To use different distance measure
---------
By default, Iris uses euclidean and Income uses cosine.
Open k_nearest_neighbors.py and include this key value pair "ratio_similarity": "cosine" to the metadata dict that is being passed to k_similar_records method.
Adding this ensure that cosine similarity is used. Remove this key value pair from the dict to use euclidean distance based similarity.