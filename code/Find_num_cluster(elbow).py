import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from kneed import KneeLocator

df = pd.read_pickle('data/standard_CLIP_feature.pkl')
X = np.vstack(df['features'].values)
pca = PCA(n_components=50)
pca_features = pca.fit_transform(X)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=10, random_state=42)
u_features = reducer.fit_transform(pca_features)

inertia = []
for k in range(1, 50):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(u_features)
    inertia.append(kmeans.inertia_)
kn = KneeLocator(range(1, 50), inertia, curve='convex', direction='decreasing')
print(f"Optimal clusters: {kn.knee}")

plt.plot(range(1, 50), inertia, 'bx-')
plt.xticks(np.arange(1, 50, 1))
plt.xlabel('number of  Cluster (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method to find the bend')
plt.show()