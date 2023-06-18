import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import random
from random import sample
from scipy.spatial import distance_matrix



class KMeans:
    
    def __init__(self, m_clusters = 2, max_iter = 1000, plot_all = False,
                 seed = None, sample = "smart_sample", repeats = 0, worst_prec = 0.05, n_rerolls = 0):
        self.m_clusters = m_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.plot_all = plot_all
        self.seed = seed
        self.sample = sample
        self.repeats = repeats
        self.worst_prec = worst_prec
        self.n_rerolls = n_rerolls
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        if self.seed:
            random.seed(self.seed)
        
        best_distortion = np.float("-inf")
        best_centroids = None    
            
        for _ in range(self.repeats+1):
            cluster_ind = np.repeat(0, X.shape[0])

            #Initate centroids and cluster members
            if self.sample == "random":
                # Assigns each centroid as a uniformly random coordinat
                for i in range(self.m_clusters):
                    c = np.array([])
                    for col in X.columns:
                        x_min = np.min(X[col])
                        x_max = np.max(X[col])
                        rand_int = random.uniform(x_min,x_max)
                        c = np.append(c,rand_int)

                    if i== 0:
                        self.centroids = np.array([c])
                    else:
                        self.centroids = np.append(self.centroids,[c], axis = 0)

            elif self.sample == "sample":
                # Sample the centroid randomly from the data
                idx = np.random.randint(X.shape[0], size=self.m_clusters)
                self.centroids = X.loc[idx].to_numpy()

            elif self.sample == "smart_sample":
                # Sample randomly from the data, but with weighted probabilities 
                # such that centroids are more likely to be spread out         
                idx = np.random.choice(range(X.shape[0]), size=1)
                centroids = X.loc[idx].to_numpy()
                while len(centroids)<self.m_clusters:
                    distances = cross_euclidean_distance(centroids, X.to_numpy())
                    prob_vec = distances.min(axis = 0)
                    prob_vec = prob_vec**2/np.sum(prob_vec**2)
                    #Note: zero proability that new centorid is allready a centroid
                    idx = np.append(idx, np.random.choice(X.shape[0], size=1, p = prob_vec)) 
                    centroids = X.loc[idx].to_numpy()

                self.centroids = centroids
    
            cross_dist = cross_euclidean_distance(X.to_numpy(), self.centroids)
            cluster_ind = np.argmin(cross_dist, axis = 1)
            
            if self.plot_all:
                z = cluster_ind
                C = self.centroids.copy()
                K = len(C)
                _, ax = plt.subplots(figsize=(5, 5), dpi=100)
                sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax);
                sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
                ax.legend().remove();
            
            
            for _ in range(self.max_iter):
                #Calculating new centroids
                for i in range(self.m_clusters):
                    X_i = X[cluster_ind == i]
                    self.centroids[i] = X_i.mean(axis = 0).to_numpy()
                
                #Assigne data points to new cluster and check if cluster assignment chenges
                cross_dist = cross_euclidean_distance(X.to_numpy(), self.centroids)
                cluster_ind_new = np.argmin(cross_dist, axis = 1)
                if not (cluster_ind_new == cluster_ind).any():
                    break
                cluster_ind = cluster_ind_new
                
                # Plot the progress
                if self.plot_all:
                    z = cluster_ind
                    C = self.centroids.copy()
                    K = len(C)
                    _, ax = plt.subplots(figsize=(5, 5), dpi=100)
                    sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax);
                    sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
                    ax.legend().remove();
                    
            # Move bad cluster to for more optimal coverage
            n_worst = int(round(X.shape[0]*self.worst_prec))
            best_reroll_centroids = self.centroids
            best_reroll_distortion = euclidean_distortion(X, cluster_ind)
            for i in range(self.n_rerolls):
                # Caculate new centroids
                for i in range(self.m_clusters):
                    X_i = X[cluster_ind == i]
                    self.centroids[i] = X_i.mean(axis = 0).to_numpy()
                
                # Find the two centroids that are closest and pick the centoid with the lowest average distance to other centroids
                centroid_dist = cross_euclidean_distance(self.centroids)
                cetorid_dist_inf = centroid_dist + np.diag(np.repeat(np.inf, centroid_dist.shape[0])) # Add inf to diag
                worst_pair = np.unravel_index((cetorid_dist_inf).argmin(), cetorid_dist_inf.shape) # Find indexes of worst pair
                worst_ind = worst_pair[0] if (np.mean(centroid_dist[worst_pair[0]])<np.mean(centroid_dist[worst_pair[1]])) else worst_pair[1]
                
                # Assign the old centroid to be the one closest to the poinst that are furthest away from the current centroids
                min_dists = np.min(cross_dist, axis = 1)
                high_dists_ind = np.argpartition(min_dists, -n_worst)[-n_worst:]
                X_high = X.loc[high_dists_ind]
                self.centroids[worst_ind] = X_high.mean(axis = 0).to_numpy()
                
                # Itterate until convergence
                for _ in range(self.max_iter):
                    #Calculating new centroids
                    for i in range(self.m_clusters):
                        X_i = X[cluster_ind == i]
                        self.centroids[i] = X_i.mean(axis = 0).to_numpy()
                    
                    #Assigne data points to new cluster and check if cluster assignment chenges
                    cross_dist = cross_euclidean_distance(X.to_numpy(), self.centroids)
                    cluster_ind_new = np.argmin(cross_dist, axis = 1)
                    if not (cluster_ind_new == cluster_ind).any():
                        break
                    cluster_ind = cluster_ind_new
                
                if self.plot_all:
                    z = cluster_ind
                    C = self.centroids.copy()
                    K = len(C)
                    _, ax = plt.subplots(figsize=(5, 5), dpi=100)
                    sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax);
                    sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
                    ax.legend().remove();
                
                distortion = euclidean_distortion(X, cluster_ind)
                if distortion<best_reroll_distortion:
                    best_reroll_distortion = distortion
                    best_reroll_centroids = self.centroids
            self.centroids = best_reroll_centroids 
            cross_dist = cross_euclidean_distance(X.to_numpy(), self.centroids)
            cluster_ind = np.argmin(cross_dist, axis = 1)
                
            distortion = euclidean_distortion(X, cluster_ind)
            if distortion<best_distortion:
                best_distortion = distortion
                best_centroids = self.centroids
        
        self.centroids = best_centroids if not best_centroids == None else self.centroids
             
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = (X-X.min())/(X.max()-X.min())
        n = X.shape[0]
        cluster_ind = np.repeat(0,n)
        
        for i in range(n):
            x = X.loc[i].to_numpy()
            min_dist = np.float("inf")
            min_cluster = 0
            for j in range(self.m_clusters):
                x_dist = euclidean_distance(x, self.centroids[j])
                if x_dist<min_dist:
                    min_dist = x_dist
                    min_cluster = j
            cluster_ind[i] = np.array(min_cluster)
        
        return cluster_ind
            
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids

        
    
    
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))