
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import umap
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator



def Calculate_cluster_DBSCAN(df,num_neighbors=15,n_comp_pca=50,n_comp_umap=10):
        pca = PCA(n_components=n_comp_pca)
        reducer = umap.UMAP(n_neighbors=num_neighbors, min_dist=0.1, n_components=n_comp_umap, random_state=42)
        
        X=np.vstack(df['features'].values)
        pca_features = pca.fit_transform(X)
        u_features = reducer.fit_transform(pca_features)
        neighbors = NearestNeighbors(n_neighbors=num_neighbors)
        neighbors_fit = neighbors.fit(u_features)
        distances, indices = neighbors_fit.kneighbors(u_features)
        # Sort distances to the k-th nearest neighbor
        distances = np.sort(distances[:, -1], axis=0)
        kneedle = KneeLocator(
            range(len(distances)),
            distances,
            curve="convex",
            direction="increasing"
        )
        eps_auto = distances[kneedle.knee]
        dbscan = DBSCAN(eps=eps_auto, min_samples=n_comp_umap+1)
        clusters_dbscan = dbscan.fit_predict(u_features)
        df['clusters_dbscan'] = clusters_dbscan
        
        n_clusters_found_C = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
        n_noise = list(clusters_dbscan).count(-1)
        percent_noise1 = (n_noise / len(clusters_dbscan)) * 100
        print(f"DBSCAN ha trovato {n_clusters_found_C} cluster "+f"\nrumore e'{percent_noise1}")
        return clusters_dbscan

if __name__ == "__main__":
    filenames=['data/standard_CLIP_feature.pkl','data/standard_DINOv2_feature.pkl','data/standard_ResNet50_feature.pkl']
    for f in filenames:
        df=pd.read_pickle(f)
        cluster=Calculate_cluster_DBSCAN(df)
        df.to_pickle(f)

'''

CLIP_df = pd.read_pickle('data/standard_CLIP_feature.pkl')
DINO_df = pd.read_pickle('data/standard_DINOv2_feature.pkl')
ResNet_df = pd.read_pickle('data/standard_ResNet50_feature.pkl')


# Extract value of features in every model，
# np.vstack()take the vector and stack it as matrix
X_CLIP = np.vstack(CLIP_df['features'].values)
X_DINO = np.vstack(DINO_df['features'].values)
X_ResNet = np.vstack(ResNet_df['features'].values)


pca = PCA(n_components=50)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=10, random_state=42)
pca_features_CLIP = pca.fit_transform(X_CLIP)
u_features_CLIP = reducer.fit_transform(pca_features_CLIP)
pca_features_DINO = pca.fit_transform(X_DINO)
u_features_DINO = reducer.fit_transform(pca_features_DINO)
pca_features_ResNet = pca.fit_transform(X_ResNet)
u_features_ResNet = reducer.fit_transform(pca_features_ResNet)
min_sampls = 11
neighbors = NearestNeighbors(n_neighbors=min_sampls)
neighbors_fit = neighbors.fit(u_features_CLIP) 
distances, indices = neighbors_fit.kneighbors(u_features_CLIP)

# Sort distances to the k-th nearest neighbor
distances = np.sort(distances[:, -1], axis=0)
kneedle = KneeLocator(
    range(len(distances)),
    distances,
    curve="convex",
    direction="increasing"
)
eps_auto = distances[kneedle.knee]



#Eps the longest distance of two embedding to be considered as neighbor,if to small get too much -1 as rumor
#the minimium number of sample to constructe a cluster, for regular of resnet it min value should be vector's dimension+1
#so instead look at vector size, try to choose a small number to prevent the rumor.
dbscan = DBSCAN(eps=eps_auto, min_samples=min_sampls)
clusters_dbscan_CLIP = dbscan.fit_predict(u_features_CLIP)
clusters_dbscan_DINO = dbscan.fit_predict(u_features_DINO)
clusters_dbscan_ResNet = dbscan.fit_predict(u_features_ResNet)


n_clusters_found_C = len(set(clusters_dbscan_CLIP)) - (1 if -1 in clusters_dbscan_CLIP else 0)
n_noise = list(clusters_dbscan_CLIP).count(-1)
percent_noise1 = (n_noise / len(clusters_dbscan_CLIP)) * 100
print(f"DBSCAN ha trovato {n_clusters_found_C} cluster in CLIP"+f"\nrumore e'{percent_noise1}")

n_clusters_found_N = len(set(clusters_dbscan_DINO)) - (1 if -1 in clusters_dbscan_DINO else 0)
n_noise = list(clusters_dbscan_DINO).count(-1)
percent_noise2 = (n_noise / len(clusters_dbscan_DINO)) * 100
print(f"DBSCAN ha trovato {n_clusters_found_N} cluster in DINO"+f"\nrumore e'{percent_noise2}")

n_clusters_found_R = len(set(clusters_dbscan_ResNet)) - (1 if -1 in clusters_dbscan_ResNet else 0)
n_noise = list(clusters_dbscan_ResNet).count(-1)
percent_noise3 = (n_noise / len(clusters_dbscan_ResNet)) * 100
print(f"DBSCAN ha trovato {n_clusters_found_R} cluster in ResNet"+f"\nrumore e'{percent_noise3}")

'''