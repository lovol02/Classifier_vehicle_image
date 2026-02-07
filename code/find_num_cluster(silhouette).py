import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap

df = pd.read_pickle('data/standard_CLIP_feature.pkl')
X = np.vstack(df['features'].values)
pca = PCA(n_components=50)
pca_features = pca.fit_transform(X)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=10, random_state=42)
u_features = reducer.fit_transform(pca_features)

silhouette_avg = []
range_n_clusters = range(2, 60) 

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(u_features)
    silhouette_avg.append(silhouette_score(u_features, cluster_labels))

plt.plot(range_n_clusters, silhouette_avg, 'ro-')
plt.xlabel('Number of Cluster')
plt.ylabel('Point of Silhouette')
plt.title('Silhouette method(find the local max)')
plt.show()