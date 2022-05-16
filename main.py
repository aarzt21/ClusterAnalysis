'''
This script defines the ClusterAnalyzer class which allows to
infer the optimal number of clusters for 3 different clustering algorithms. 

Author: Alex Arzt
Date: 11 May 2022
'''
from typing import List, Type
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

import scipy
import scipy.cluster.vq
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean
import seaborn as sns

class ClusterAnalyzer:

    def __init__(self, k_min = 1, k_max = 9) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.X = None 
    

    def fit(self, X) -> None:
        """Pass in the data

        Args:
            X (ndarry): all the features in numeric form
        """
        if isinstance(X, np.ndarray):
            self.X = X
        else: 
            raise TypeError("Feature matrix must be of type nump.ndarray with all numeric features!")
        

    def _get_means(self, k, preds) -> np.ndarray:

        means = np.zeros(shape=(k,self.X.shape[1]))
        X2 = np.copy(np.hstack([self.X, preds]))
        for label in np.unique(preds):
            cluster = X2[X2[:,-1] == label][:,0:-1]
            means[label,:] = np.mean(cluster, axis=0)
        return means

    
    def _calculate_WSS(self, model, data) -> list[float]:

        wss = []
        for k in range(1, self.k_max+1):
            if model == "kmeans":
                kmeans = KMeans(n_clusters = k).fit(data)
                centroids = kmeans.cluster_centers_
                pred_clusters = kmeans.predict(data)
            if model == "kmedoids":
                kmediods = KMedoids(n_clusters = k).fit(data)
                centroids = kmediods.cluster_centers_
                pred_clusters = kmediods.predict(data)
            if model == 'agg':
                agg = AgglomerativeClustering(n_clusters=k).fit(data)
                pred_clusters = agg.fit_predict(data).reshape((len(data),1))
                centroids = self._get_means(k,pred_clusters.reshape((len(data),1)))

            curr_wss = 0
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(self.X)):
                curr_center = centroids[pred_clusters[i]]
                curr_wss += distance.euclidean(data[i,:], curr_center)
            wss.append(curr_wss)
        return wss


    def plot_elbow(self) -> None:
        """
        plots the elbow heuristic for all models
        """
        if isinstance(self.X, np.ndarray):
            wss_kmeans = self._calculate_WSS(model = 'kmeans', data=self.X)
            wss_kmedoids = self._calculate_WSS(model = 'kmedoids', data=self.X)
            wss_agg = self._calculate_WSS(model = 'agg', data=self.X)
            plt.figure(figsize=(20,12))
            plt.plot(np.arange(1,self.k_max+1), wss_kmeans, 'bx-', label='K-Means')
            plt.xlabel('k')
            plt.ylabel('WSS')
            plt.plot(np.arange(1,self.k_max+1), wss_kmedoids, 'rx-', label='K-Medoids')
            plt.plot(np.arange(1,self.k_max+1), wss_agg, 'gx-', label='Agg Clustering')
            plt.title('The Elbow Heuristic Showing the Optimal k')
            plt.legend()
            sns.set_theme()
            plt.show()
        else: 
            raise TypeError("You have not supplied any data! Use the fit method!")

    
    def _get_silhouette(self, model) -> list[float]:

        sihl_coefs = []
        for k in range(2, self.k_max+1):
            if model == "kmeans":
                kmeans = KMeans(n_clusters = k).fit(self.X)
                pred_clusters = kmeans.predict(self.X)
            if model == "kmedoids":
                kmediods = KMedoids(n_clusters = k).fit(self.X)
                pred_clusters = kmediods.predict(self.X)
            if model == 'agg':
                agg = AgglomerativeClustering(n_clusters=k).fit(self.X)
                pred_clusters = agg.fit_predict(self.X)
            sihl_coefs.append(silhouette_score(self.X, labels= pred_clusters))
        return sihl_coefs


    def plot_silhouette_coefs(self) -> None:
        """
        plots the silhouette coefficients for all models
        """
        if isinstance(self.X, np.ndarray):
            sc_kmeans = self._get_silhouette(model = 'kmeans')
            sc_kmedoids = self._get_silhouette(model = 'kmedoids')
            sc_agg = self._get_silhouette(model = 'agg')
            plt.figure(figsize=(20,12))
            plt.plot(np.arange(2,self.k_max+1), sc_kmeans, 'bx-', label='K-Means')
            plt.xlabel('k')
            plt.ylabel('WSS')
            plt.plot(np.arange(2,self.k_max+1), sc_kmedoids, 'rx-', label='K-Medoids')
            plt.plot(np.arange(2,self.k_max+1), sc_agg, 'gx-', label='Agg Clustering')
            plt.title('The Silhhouette Coefficient Showing the Optimal k')
            plt.legend()
            sns.set_theme()
            plt.show()
        else: 
            raise TypeError("You have not supplied any data! Use the fit method!")


    
    def _get_gap_stats(self, model) -> list:
        wss = self._calculate_WSS(model=model, data=self.X)
        wss_rand = self._calculate_WSS(model=model, data=np.random.random((self.X.shape[0], self.X.shape[1])) * 2 - 1)
        
        
        return list(np.log(np.array(wss_rand))-np.log(np.array(wss)))


    def plot_gap_stat(self):
        """
        plots the gapstat for all models
        """
        if isinstance(self.X, np.ndarray):
            gs_kmeans = self._get_gap_stats(model = 'kmeans')
            gs_kmedoids = self._get_gap_stats(model = 'kmedoids')
            gs_agg = self._get_gap_stats(model = 'agg')
            plt.figure(figsize=(20,12))
            plt.plot(np.arange(1,self.k_max+1), gs_kmeans, 'bx-', label='K-Means')
            plt.xlabel('k')
            plt.ylabel('WSS')
            plt.plot(np.arange(1,self.k_max+1), gs_kmedoids, 'rx-', label='K-Medoids')
            plt.plot(np.arange(1,self.k_max+1), gs_agg, 'gx-', label='Agg Clustering')
            plt.title('The Gap Statistic Showing the Optimal k')
            plt.legend()
            sns.set_theme()
            plt.show()
        else: 
            raise TypeError("You have not supplied any data! Use the fit method!")



    def gimme_all(self) -> None:
        """generates all plots in one swoop
        """
        sns.set_theme()
        self.plot_elbow()
        self.plot_silhouette_coefs()
        self.plot_gap_stat()

    def get_report(self, title) -> None:
        """generates a png graphic with all the plots

        Args:
            title (str): [your_title].png

        """

        if not isinstance(self.X, np.ndarray):
            raise TypeError("You have not supplied any data! Use the fit method!")

        sns.set_theme()
        figure, axis = plt.subplots(3, 3, figsize=(15,15))
        wss_kmeans = self._calculate_WSS(model = 'kmeans', data=self.X)
        wss_kmedoids = self._calculate_WSS(model = 'kmedoids', data=self.X)
        wss_agg = self._calculate_WSS(model = 'agg', data=self.X)
        
        sc_kmeans = self._get_silhouette(model = 'kmeans')
        sc_kmedoids = self._get_silhouette(model = 'kmedoids')
        sc_agg = self._get_silhouette(model = 'agg')

        gs_kmeans = self._get_gap_stats(model = 'kmeans')
        gs_kmedoids = self._get_gap_stats(model = 'kmedoids')
        gs_agg = self._get_gap_stats(model = 'agg')


        # elbow 
        axis[0, 0].plot(np.arange(1,self.k_max+1), wss_kmeans)
        axis[0, 0].set_title("Elbow Heuristic Kmeans")
        # elbow Kmedoids
        axis[0, 1].plot(np.arange(1,self.k_max+1), wss_kmedoids)
        axis[0, 1].set_title("Elbow Heuristic Kmedoids")
        # elbow Agg
        axis[0, 2].plot(np.arange(1,self.k_max+1), wss_agg)
        axis[0, 2].set_title("Elbow Heuristic Agglomerative Clustering")

        # sc 
        axis[1, 0].plot(range(2, self.k_max+1), sc_kmeans)
        axis[1, 0].set_title("Silhouette Coefficients Kmeans")

        axis[1, 1].plot(range(2, self.k_max+1), sc_kmedoids)
        axis[1, 1].set_title("Silhouette Coefficients Kmedoids")

        axis[1, 2].plot(range(2, self.k_max+1), sc_agg)
        axis[1, 2].set_title("Silhouette Coefficients Agglomerative Clustering")

        # gap stat
        axis[2, 0].plot(range(1, self.k_max+1), gs_kmeans)
        axis[2, 0].set_title("Gap Stats Kmeans")

        axis[2, 1].plot(range(1, self.k_max+1), gs_kmedoids)
        axis[2, 1].set_title("Gap Stats Kmedoids")

        axis[2, 2].plot(range(1, self.k_max+1), gs_agg)
        axis[2, 2].set_title("Gap Stats Agglomerative Clustering")

        plt.savefig(title)


# testing

if __name__ == "__main__":
    from copy import deepcopy
    from ctypes.wintypes import MSG
    from turtle import color
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Size of dataset to be generated. The final size is 4 * data_size
    data_size = 1000
    num_iters = 50
    num_clusters = 4

    # sample from Gaussians 
    data1 = np.random.normal((5,5,5), (4, 4, 4), (data_size,3))
    data2 = np.random.normal((4,20,20), (3,3,3), (data_size, 3))
    data3 = np.random.normal((25, 20, 5), (5, 5, 5), (data_size,3))
    data4 = np.random.normal((30, 30, 30), (5, 5, 5), (data_size,3))

    # Combine the data to create the final dataset
    data = np.concatenate((data1,data2, data3, data4), axis = 0)


    import main
    analyzer = ClusterAnalyzer()
    analyzer.fit(data)
    #analyzer.plot_elbow()
    #analyzer.plot_silhouette_coefs()
    analyzer.gimme_all()



    """ rand = np.random.random((data.shape[0], data.shape[1])) * 2 - 1

    # Plotting a 3d plot using matplotlib to visualize the data points
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rand[:,0], rand[:,1], rand[:,2])
    plt.show() """


    #analyzer.plot_gap_stat()

    analyzer.get_report('report.png')