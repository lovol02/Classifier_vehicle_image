import numpy as np 
import umap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import pairwise_distances_argmin_min
from pathlib import Path
import shutil
from PIL import Image
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.decomposition import PCA
import umap
import cv2
from ResNet50 import Resnet50_model_config
from CLIP import CLIP_model_config
from DINOv2 import DINOV2_model_config
import json

def embedding_DimensionReduction2D(embedding,n_component=50,n_neighbors=15,n_components=2):
    pca = PCA(n_components=n_component)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(embedding)
    u_features = reducer.fit_transform(pca_features)
    return u_features

# This function can be migliorized
def UMAP_visualizzation(df,df1,df2,dfile):
    image_embeddings = np.stack(df['features'].values)
    image_embeddings1 = np.stack(df1['features'].values)
    image_embeddings2 = np.stack(df2['features'].values)
    embedding_2d=embedding_DimensionReduction2D(image_embeddings)
    embedding_2d1=embedding_DimensionReduction2D(image_embeddings1)
    embedding_2d2=embedding_DimensionReduction2D(image_embeddings2)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df1['year'] = pd.to_numeric(df1['year'], errors='coerce')
    df2['year'] = pd.to_numeric(df2['year'], errors='coerce')
    
    plt.figure(1, figsize=(10, 7))
    scatter=plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=df['clusters_kmeans'], cmap='turbo', s=1 )
    plt.suptitle(df.name)
    plt.title("Visualizzazione Clustering: K-Means")
    plt.legend(
        *scatter.legend_elements(num=None), 
        title="Cluster IDs", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df.name+"_KMeans.png")
    
    
    plt.figure(2, figsize=(10, 7))
    scatter1 = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=df['clusters_dbscan'], cmap='turbo', s=1 )
    plt.suptitle(df.name)
    plt.title("Visualizzazione Clustering: DBSCAN")
    plt.legend(
        *scatter1.legend_elements(num=None), 
        title="Cluster IDs", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df.name+"_DBSCAN.png")
    
    plt.figure(7, figsize=(10, 7))
    scatter=plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=df['year'], cmap='turbo', s=1 )
    plt.suptitle(df.name)
    plt.title("Visualizzazione years evolution")
    plt.legend(
        *scatter.legend_elements(num=None), 
        title="years", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df.name+"years.png")

    plt.figure(3, figsize=(10, 7))
    scatter2=plt.scatter(embedding_2d1[:, 0], embedding_2d1[:, 1], c=df1['clusters_kmeans'], cmap='turbo', s=1 )
    plt.suptitle(df1.name)
    plt.title("Visualizzazione Clustering: K-Means")
    plt.legend(
        *scatter2.legend_elements(num=None), 
        title="Cluster IDs", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df1.name+"_Kmeans.png")
    
    plt.figure(4, figsize=(10, 7))
    scatter3 = plt.scatter(embedding_2d1[:, 0], embedding_2d1[:, 1], c=df1['clusters_dbscan'], cmap='turbo', s=1 )
    plt.suptitle(df1.name)
    plt.title("Visualizzazione Clustering: DBSCAN")
    plt.legend(
        *scatter3.legend_elements(num=None), 
        title="Cluster IDs", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df1.name+"_DBSCAN.png")
    
    plt.figure(8, figsize=(10, 7))
    scatter=plt.scatter(embedding_2d1[:, 0], embedding_2d1[:, 1], c=df1['year'], cmap='turbo', s=1 )
    plt.suptitle(df1.name)
    plt.title("Visualizzazione years evolution")
    plt.legend(
        *scatter.legend_elements(num=None), 
        title="years", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df1.name+"years.png")
    
    plt.figure(5, figsize=(10, 7))
    scatter4=plt.scatter(embedding_2d2[:, 0], embedding_2d2[:, 1], c=df2['clusters_kmeans'], cmap='turbo', s=1 )
    plt.suptitle(df2.name)
    plt.title("Visualizzazione Clustering: K-Means")
    plt.legend(
        *scatter4.legend_elements(num=None), 
        title="Cluster IDs", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df2.name+"_KMeans.png")
    
    plt.figure(6, figsize=(10, 7))
    scatter5 = plt.scatter(embedding_2d2[:, 0], embedding_2d2[:, 1], c=df2['clusters_dbscan'], cmap='turbo', s=1 )
    plt.suptitle(df2.name)
    plt.title("Visualizzazione Clustering: DBSCAN")
    plt.legend(
        *scatter5.legend_elements(num=None), 
        title="Cluster IDs", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df2.name+"_DBSCAN.png")

    plt.figure(9, figsize=(10, 7))
    scatter=plt.scatter(embedding_2d2[:, 0], embedding_2d2[:, 1], c=df2['year'], cmap='turbo', s=1 )
    plt.suptitle(df2.name)
    plt.title("Visualizzazione years evolution")
    plt.legend(
        *scatter.legend_elements(num=None), 
        title="years", 
        loc="upper right",
        bbox_to_anchor=(1.15, 1) # This moves the legend outside the plot
    )
    plt.savefig(dfile+df2.name+"years.png")
    plt.show()
    temporal_analysis1 = df.groupby('clusters_kmeans')['year'].agg(['mean', 'median', 'std']).sort_values(by='mean')
    temporal_analysis2 = df.groupby('clusters_dbscan')['year'].agg(['mean', 'median', 'std']).sort_values(by='mean')
    temporal_analysis3 = df1.groupby('clusters_kmeans')['year'].agg(['mean', 'median', 'std']).sort_values(by='mean')
    temporal_analysis4 = df1.groupby('clusters_dbscan')['year'].agg(['mean', 'median', 'std']).sort_values(by='mean')
    temporal_analysis5 = df2.groupby('clusters_kmeans')['year'].agg(['mean', 'median', 'std']).sort_values(by='mean')
    temporal_analysis6 = df2.groupby('clusters_dbscan')['year'].agg(['mean', 'median', 'std']).sort_values(by='mean')
    data_to_save = {
        "clusters_kmeans": temporal_analysis1.to_dict(),
        "clusters_dbscan": temporal_analysis2.to_dict(),
        "clusters_kmeans": temporal_analysis3.to_dict(),
        "clusters_dbscan": temporal_analysis4.to_dict(),
        "clusters_kmeans": temporal_analysis5.to_dict(),
        "clusters_dbscan": temporal_analysis6.to_dict(),
    }
    with open(dfile+'temporal_analysis.json', 'w') as f:
        json.dump(data_to_save, f, indent=4)

class SimilarityTarget:
    def __init__(self, centroid):
        self.centroid = centroid

    def __call__(self, model_output):
        # Determine if it's CNN or ViT based on dimensions
        if len(model_output.shape) == 4: # In total 4 spartial feature map, [Batch, Channels, Height, Width]
            z = torch.nn.functional.adaptive_avg_pool2d(model_output, 1).flatten(1)
        elif model_output.dim() == 3: # In total 3 element at last layer, [Batch, Tokens, Embedding_Dim]
            z = model_output[:, 0, :]
        else:
             z = model_output

        # Cosine Similarity Calculation
        z_norm = torch.nn.functional.normalize(z, dim=-1)
        mu_norm = torch.nn.functional.normalize(self.centroid, dim=-1)
        
        # Return scalar sum for backprop
        return (z_norm * mu_norm).sum()

def vit_reshape_transform(tensor):
    # tensor shape is usually [Batch, Tokens, Channels]
    # 1. Determine how many non-patch tokens there are
    # For DINOv2 with registers: 1 (CLS) + 4 (Reg) = 5
    # For CLIP: 1 (CLS) = 1
    num_tokens = tensor.shape[1]
    if num_tokens == 257: # 224/14 = 16. 16*16 = 256 + 1 CLS
        start_index = 1
    elif num_tokens == 261: # 256 patches + 1 CLS + 4 Registers
        start_index = 5
    else:
        # Fallback for other resolutions
        start_index = 1 

    patch_tokens = tensor[:, start_index:, :] 
    
    # Calculate grid size (e.g., 14 for 196 patches or 16 for 256 patches)
    num_patches = patch_tokens.shape[1]
    grid_size = int(num_patches**0.5) 
    
    result = patch_tokens.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
          
def grad_cam(centroids,dic,df,sfolder,dfolder):
    reshape_func = None
    if df.name == "ResNet50":
        device,model,transform=Resnet50_model_config()
        # The target layer for ResNet50 is the layer before average pool layer
        target_layers = [model[7][-1]]
    elif df.name == "DINOv2":
        device,model,transform= DINOV2_model_config()
        target_layers = [model.blocks[-1].norm1]
        reshape_func = vit_reshape_transform
    else:
        device,model,transform,tokenizer = CLIP_model_config()
        target_layers = [model.visual.transformer.resblocks[-2]]
        reshape_func = vit_reshape_transform
    # Define grad-cam
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_func)
            
    for k,v in dic.items():
        centroid=centroids[k]
        for idx in v:
            fpath=dfolder+str(k)+"/"
            path = Path(fpath)
            path.mkdir(parents=True, exist_ok=True)
            image=sfolder+df.loc[idx, 'file_name']
            dstfile=fpath+"heatmap_"+df.loc[idx, 'file_name']
 
            centroid_tensor = torch.from_numpy(centroid).float().to(device)
            img = Image.open(image).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
    
            # Process with grad-cam
            targets = [SimilarityTarget(centroid_tensor)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

            # Overlay the heatmap on the image
            img_as_numpy = np.array(img)
            img_float = np.float32(img_as_numpy) / 255.0
            img_float = cv2.resize(img_float, (224, 224))
            visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            cv2.imwrite(dstfile, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
    

def get_cluster_prototypes(df, cluster_col='clusters_kmeans', n_proto=3):
    """
    Finds the images closest to the center of each cluster.
    """
    proto_indices = {}
    centroids = {}
    unique_clusters = sorted(df[cluster_col].unique())
    
    
    for cluster in unique_clusters:
        if cluster == -1: continue # Skip DBSCAN noise
        
        # 1. Filter data for this specific cluster
        cluster_data = df[df[cluster_col] == cluster]
        cluster_embeddings = np.array(cluster_data['features'].tolist())
        #keeping the index
        cluster_df_indices = cluster_data.index.tolist()

        
        # 2. Calculate the Centroid (Mean) and reshape to 1 row, otherwise maybe n rows with 1 column
        centroid = cluster_embeddings.mean(axis=0).reshape(1, -1)
        
        # 3. Find the points closest to the centroid
        # We calculate euclidean distances from all points in cluster to the centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # 4. Get the indices of the 'n_proto' smallest distances
        # argsort sorts the positions (indices) of those values, the return are indexs that has smallest value
        closest_local_indices = np.argsort(distances)[:n_proto]
        
        # 5. Map back to the original dataframe indices
        proto_indices[cluster] = [cluster_df_indices[i] for i in closest_local_indices]
        centroids[cluster] = centroid
    return proto_indices,centroids

def show_imgs(path, dic,df):
    col = len(next(iter(dic.values())))
    row = len(dic)
    fig, axes = plt.subplots(row, col, figsize=(15, 4*row/2))
    axes = axes.flatten()
    count=0
    for k,v in dic.items():
        for idx in v:
            #print(df.loc[idx, 'file_name'])
            CPath=path+df.loc[idx, 'file_name']
            print(CPath)
            try:
                img = mpimg.imread(CPath)
                axes[count].imshow(img)
                axes[count].set_title("of cluster "+ str(k))
                axes[count].axis('off')

            except Exception as e:
                print("file not found")
                axes[count].text(0.5, 0.5, "Img Not Found", ha='center')
                axes[count].axis('off')

            count+=1
    # Its sole job is to automatically adjust the padding between 
    # and around your subplots so that nothing overlaps.
    plt.tight_layout()
    plt.show()

def copy_img(sfolder,dfolder,dic,df):
    for k,v in dic.items():
        for idx in v:
            #print(df.loc[idx, 'file_name'])
            fpath=dfolder+str(k)+"/"
            path = Path(fpath)
            path.mkdir(parents=True, exist_ok=True)
            srcfile=sfolder+df.loc[idx, 'file_name']
            dstfile=fpath+df.loc[idx, 'file_name']
            print(dstfile)
            shutil.copy(srcfile, dstfile)
    

    
def extract_imagePath(dic,sfolder,df):
    paths=[]
    for k,v in dic.items():
        for idx in v:
            paths.append(sfolder+df.loc[idx, 'file_name'])
    return paths

def prototype_extract(dfs,img_folder,dest_folder,Methods_clustering):
    for method in Methods_clustering:
        for df in dfs:
            if method == "Kmeans":
                col="clusters_kmeans"
            else:
                col="clusters_dbscan"
            proto_indices, centroids=get_cluster_prototypes(df,col)
            dest_folder1=dest_folder+method+"/"+df.name+"/"
            copy_img(img_folder,dest_folder1,proto_indices, df)
            grad_cam(centroids,proto_indices,df,img_folder,dest_folder1)


               

# Just exclude following code, those code are used for the test
if __name__ == "__main__":
    imgs_folder="dataset/cars_train/cars_train/"
    dest_folder="temp/"
    prompt=["sporty aggressive", "elegant luxury", "rugged off-road", "futuristic", "vintage"]
    filenames=['data/standard_CLIP_feature.pkl','data/standard_DINOv2_feature.pkl','data/standard_ResNet50_feature.pkl']
    img=".\\temp\Kmeans\Restnet50\\0\\00377.jpg"
    grad_cam(img,0)
    '''
    df1=pd.read_pickle('data/standard_CLIP_feature.pkl')
    df1.name="CLIP"
    df2=pd.read_pickle('data/standard_DINOv2_feature.pkl')
    df2.name="DINOV2"
    df3=pd.read_pickle('data/standard_ResNet50_feature.pkl')
    df3.name="ResNet"
    #UMAP_visualizzation(df,df1,df2)
    proto_indices1 = get_cluster_prototypes(df1)
    proto_indices2 = get_cluster_prototypes(df2)
    proto_indices3 = get_cluster_prototypes(df3)
    #print([v for k,v in proto_indices.items() ])
    #show_imgs(imgs_folder,proto_indices,df2)
    
    paths=extract_imagePath(proto_indices2,imgs_folder,df2)
    attentionMap(paths)
    
    dest_folder1=dest_folder+"Kmeans/"+"CLIP/"
    dest_folder2=dest_folder+"Kmeans/"+"DINOv2/"
    dest_folder3=dest_folder+"Kmeans/"+"Restnet50/"
    
    copy_img(imgs_folder,dest_folder1,proto_indices1, df1)
    copy_img(imgs_folder,dest_folder2,proto_indices2, df2)
    copy_img(imgs_folder,dest_folder3,proto_indices3, df3)
    
    proto_indices1 = get_cluster_prototypes(df1,"clusters_dbscan")
    proto_indices2 = get_cluster_prototypes(df2,"clusters_dbscan")
    proto_indices3 = get_cluster_prototypes(df3,"clusters_dbscan")
    dest_folder1=dest_folder+"DBSCAN/"+"CLIP/"
    dest_folder2=dest_folder+"DBSCAN/"+"DINOv2/"
    dest_folder3=dest_folder+"DBSCAN/"+"Restnet50/"
    
    copy_img(imgs_folder,dest_folder1,proto_indices1, df1)
    copy_img(imgs_folder,dest_folder2,proto_indices2, df2)
    copy_img(imgs_folder,dest_folder3,proto_indices3, df3)
    '''
     
        