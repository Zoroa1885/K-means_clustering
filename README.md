# K-means_clustering
K-means Clustering algorithm for unsupervised classification. A few scheams have been implemented to improve performance. These consist of:
1. Sampling centroids from the date sequantialy, each from a weighted distribution. Data points closer to centorids already sampled have a lower probability to be sampled.
2. Rerunning the algorithm several times and saving the best cluster assignment.
3. After convergence, moving centroids with bad coverage to the mean of the datapoints that is the furthest away form the centroids of the cluster they are assigned to. 
