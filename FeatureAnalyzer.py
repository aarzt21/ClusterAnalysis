from hashlib import blake2b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats



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

    def get_feature_analyis(self, feature, cluster) -> None:
        data = np.hstack([self.X, self.y.reshape((len(self.y),1))])
        ref = data[data[:,-1] == cluster, 0:-1]
        rest = data[data[:,-1] != cluster, 0:-1]
        mean_ref = np.mean(ref[:,feature])
        mean_rest = np.mean(rest[:,feature])
        tt_pval = stats.ttest_ind(ref[:,feature], rest[:,feature], equal_var=False)[1]
        
        sns.set_theme()
        plt.figure()
        plt.hist(ref[:,feature],density = True, bins = 20, alpha=0.9, color='violet', label= "Cluster "+str(cluster))
        plt.axvline(mean_ref, color='blue', alpha=1)
        plt.hist(rest[:,feature],density = True, bins = 20, alpha=0.4, color='grey', label = "Rest")
        plt.axvline(mean_rest, color='black', alpha=0.5)
        plt.legend()
        plt.title("Feature " + str(feature))
        plt.annotate('T-Test P-Value: ' + str(tt_pval) , xy=(0.01, 0.95), xycoords='axes fraction', color='red')
        plt.show()

        











































    
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

    feat_analyzer.get_feature_analyis(cluster=2, feature=0)
