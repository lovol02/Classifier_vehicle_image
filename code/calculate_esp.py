# %%
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
from kneed import KneeLocator
from matplotlib import plt
# %%
df = pd.read_pickle('data/standard_ResNet50_feature.pkl')
X = np.vstack(df['features'].values)
pca = PCA(n_components=50)
pca_features = pca.fit_transform(X)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=10, random_state=42)
u_features = reducer.fit_transform(pca_features)

# %%

# 1. Use your 10D UMAP features
# n_neighbors is usually set to the same as min_samples (e.g., 10)
min_samples = 11
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(u_features) # Your 10D UMAP features
distances, indices = neighbors_fit.kneighbors(u_features)

# 2. Sort distances to the k-th nearest neighbor
distances = np.sort(distances[:, -1], axis=0)

kneedle = KneeLocator(
    range(len(distances)),
    distances,
    curve="convex",
    direction="increasing"
)
eps_auto = distances[kneedle.knee]
print(eps_auto)



# 3. Plot to find the "elbow"
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title("K-Distance Plot for Epsilon Selection")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
plt.grid(True)
plt.show()


# %%
