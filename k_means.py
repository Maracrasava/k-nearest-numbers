import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot
from matplotlib import style

digits = load_digits()
# Scaling entries in each data matrix
data = scale(digits.data)

y = digits.target

# Choosing k centers

k = len(np.unique(y))

# Dimensions of data matrix
samples, features = data.shape


# Different scorings
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             # checks if each cluster contains only members of single class. ranges from 0-1 with 1 being the best score
             metrics.homogeneity_score(y, estimator.labels_),
             # checks if all members of a given class are assigned to the same cluster (ranges from 0-1 again)
             metrics.completeness_score(y, estimator.labels_),
             # harmonic mean of homo. and complet. (ranges from 0-1 again)
             metrics.v_measure_score(y, estimator.labels_),
             # similarity of the actual values and their predictions
             metrics.adjusted_rand_score(y, estimator.labels_),
             #
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             # measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation)
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


hyp = KMeans(init="k-means++", n_clusters=k, n_init=15)

bench_k_means(hyp, "1", data)




