import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
from kneed import KneeLocator

def Calculate_cluster_Kmeans(df,num_neighbors=15,n_comp_pca=50,n_comp_umap=10):
    X = np.vstack(df['features'].values)
    #Use PCA and UMAP to reduce the dimension of the data vector
    pca = PCA(n_components=n_comp_pca)
    reducer = umap.UMAP(n_neighbors=num_neighbors, min_dist=0.1, n_components=n_comp_umap, random_state=42)
    pca_features = pca.fit_transform(X)
    u_features = reducer.fit_transform(pca_features)
    #Use elbow instead silhouettes to predict the quantity of cluster
    inertia = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(u_features)
        inertia.append(kmeans.inertia_)
    #Use a KneeLocator to automatically locate the best number of cluster
    kn = KneeLocator(range(1, 50), inertia, curve='convex', direction='decreasing')
    print(f"Optimal clusters: {kn.knee}")
    kmeans = KMeans(n_clusters=kn.knee, init='k-means++', random_state=0, n_init=10)
    clusters_kmeans = kmeans.fit_predict(u_features)
    '''
    for elem in clusters_kmeans:
        print(elem)
    '''
    df['clusters_kmeans']=clusters_kmeans
    return clusters_kmeans

if __name__ == "__main__":
    filenames=['data/standard_CLIP_feature.pkl','data/standard_DINOv2_feature.pkl','data/standard_ResNet50_feature.pkl']
    for f in filenames:
        df=pd.read_pickle(f)
        cluster=Calculate_cluster_Kmeans(df)
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
pca_features_clip = pca.fit_transform(X_CLIP)
u_features_clip = reducer.fit_transform(pca_features_clip)
pca_features_dino = pca.fit_transform(X_DINO)
u_features_dino = reducer.fit_transform(pca_features_dino)
#We choose n_cluster as 8, random_state to fix the starting postion of cluster
#init='k-means++' indicate use algorithm k-means++ to choose the initial position
#n_init=10 run this algorithm 10 times 
kmeans = KMeans(n_clusters=26, init='k-means++', random_state=0, n_init=10)


clusters_kmeans_CLIP = kmeans.fit_predict(u_features_clip)
clusters_kmeans_DINO = kmeans.fit_predict(u_features_dino)
#clusters_kmeans_ResNet = kmeans.fit_predict(X_ResNet)
CLIP_df['cluster_CLIP'] = clusters_kmeans_CLIP
DINO_df['cluster_DINO'] = clusters_kmeans_DINO
#ResNet_df['cluster_ResNet'] = clusters_kmeans_ResNet

CLIP_df.to_pickle('data/CLIP_featureWith26meansCluster.pkl')
DINO_df.to_pickle('data/DINOv2_featureWith26meansCluster.pkl')
#ResNet_df.to_pickle('data/ResNet50_featureWithKmeansCluster.pkl')

print(CLIP_df[CLIP_df['cluster_CLIP'] == 0]['file_name'].head(10))
'''