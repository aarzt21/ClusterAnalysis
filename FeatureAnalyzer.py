from hashlib import blake2b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages



class FeatureAnalyzer:
    
    def __init__(self) -> None:
        self.X = None
        self.y = None
        self.model = None

    def fit(self, data, labels, model) -> None:
        """Fits the analyzer with the Clustered Data

        Args:
            data (numpy.ndarray): n x p matrix with the feautures
            labels (numpy.array): n x 1 array with the labels
            model (str): the name of the model use to produce the clustering
        """
        self.X = data
        self.y = labels
        self.model = model

    def _top_features_per_cluster(self, cluster, top=2) -> None:
        ks_stats = []
        ks_pvals = []
        for feature in range(self.X.shape[1]):
            data = np.hstack([self.X, self.y.reshape((len(self.y),1))])
            ref = data[data[:,-1] == cluster, 0:-1]
            rest = data[data[:,-1] != cluster, 0:-1]
            ks_stats.append(stats.ks_2samp(data1=ref[:,feature], data2=rest[:,feature])[0])
            ks_pvals.append(stats.ks_2samp(data1=ref[:,feature], data2=rest[:,feature])[1])

        top_feats_idx = np.argsort(ks_stats)[-top:]
        plt.figure(figsize=(18,18))
        plot_idx = 1

        for feature in top_feats_idx:
            data = np.hstack([self.X, self.y.reshape((len(self.y),1))])
            ref = data[data[:,-1] == cluster, 0:-1]
            rest = data[data[:,-1] != cluster, 0:-1]
            mean_ref = np.mean(ref[:,feature])
            mean_rest = np.mean(rest[:,feature])
            sns.set_theme(style='dark')
            plt.subplot(3, 3, plot_idx)
            plt.hist(ref[:,feature],density = True, bins = 20, alpha=0.9, color='violet', label= "Cluster "+str(cluster))
            plt.axvline(mean_ref, color='blue', alpha=1)
            plt.hist(rest[:,feature],density = True, bins = 20, alpha=0.4, color='grey', label = "Rest")
            plt.axvline(mean_rest, color='black', alpha=0.5)
            plt.legend()
            plt.title("Feature " + str(feature))
            plt.annotate('KS-Test P-Value: ' + str(round(ks_pvals[feature],3)) , xy=(0.01, 1), xycoords='axes fraction', color='darkred')
            plot_idx += 1

        plt.show()

    def get_feature_analyis(self, top) -> None:
        """takes a certain cluster c, computes the 2 Sample Kolomogorow-Smirnov
            statistic for each feature of the cluster and the "rest", and plots the 
            top features that have the largest test statistic; i.e. the top features
            that make cluster c differ from the other clusters

        Args:
            top (int): e.g. 5 would be the top 5 feautures
        """
        for cluster in np.unique(self.y):
            self._top_features_per_cluster(cluster=cluster, top=top)






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
    preds = KMeans(n_clusters=4).fit_predict(data)

    feat_analyzer = FeatureAnalyzer()
    feat_analyzer.fit(data = data, labels = preds, model = 0)

    bla = np.hstack([feat_analyzer.X, feat_analyzer.y.reshape((len(feat_analyzer.y),1))])

    bla1 = bla[bla[:,-1]==1,:-1]
    bla2 = bla[bla[:,-1]!=1,:-1]

    feat_analyzer.get_feature_analyis(top=3)
