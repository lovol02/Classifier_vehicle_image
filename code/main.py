
# %%

import os
from datetime import datetime
import pandas as pd
from metaData_process import dataframe
from CLIP import CLIP_model_config,All_SoftLabel_and_Features_CLIP
from DINOv2 import All_Features_DINOv2,DINOV2_model_config
from ResNet50 import Resnet50_model_config,All_Features_ResNet50
from Cluster_Kmeans import Calculate_cluster_Kmeans
from Cluster_DBSCAN import Calculate_cluster_DBSCAN

import visualization
import importlib
importlib.reload(visualization)

from comparison import summarize_comparison
from visualization import UMAP_visualizzation,prototype_extract
# %%
def main():
    # %%
    #prompt=["sporty aggressive", "elegant luxury", "rugged off-road", "futuristic", "vintage"]
    prompt=["a photo of a sporty aggressive style car", "a photo of a elegant luxury style car", "a photo of a rugged off-road style car", "a photo of a futuristic style car", "a photo of a vintage style car"]
    #A modification about the metadata 
    meta_file = "dataset/car_devkit/devkit/cars_meta.mat"
    anno_file = "dataset/car_devkit/devkit/cars_train_annos.mat"
    train_folder = "dataset/cars_train/images_small"
    multi_word_brands = ["Aston Martin", "Land Rover","Alfa Romeo","General Motors","Tata Motors"]
    
    #metadata='data/standard_cars_metadata.pkl'
    image_dir="dataset/cars_train/cars_train/"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"instance_{timestamp}/"
    os.makedirs(folder_name, exist_ok=True)
    dataFolder = folder_name+"data/"
    
    prototypeFolder = folder_name+"prototype/"
    os.makedirs(dataFolder, exist_ok=True)
    os.makedirs(prototypeFolder , exist_ok=True)
    
    # %%
    Clip_device, Clip_model, Clip_preprocess, Clip_tokenizer= CLIP_model_config()
    Dino_device,dinov2_model,Dino_transform = DINOV2_model_config()
    Res_device,resnet50,Res_transform = Resnet50_model_config()
    
    df_metadata=dataframe(meta_file,anno_file,multi_word_brands)
    
    # Point 1 and softlabel
    # %%
    # Compute Clip features and softlabel
    Clip_softLabel,Clip_features=All_SoftLabel_and_Features_CLIP(df_metadata,prompt,image_dir, Clip_device, Clip_model, Clip_preprocess, Clip_tokenizer)
    SoftLabel_df = pd.DataFrame(Clip_softLabel)
    SoftLabel_df.name="SoftLabel"
    CLIP_features=pd.DataFrame(Clip_features)
    CLIP_features.name="CLIP"
    # %%
    # Compute DINOv2 features
    DINO_features = All_Features_DINOv2(df_metadata,image_dir,Dino_device,dinov2_model,Dino_transform)
    DINOv2_features=pd.DataFrame(DINO_features)
    DINOv2_features.name="DINOv2"
    # %%
    # Compute ResNet features
    ResNet_features=All_Features_ResNet50(df_metadata, image_dir, Res_device, resnet50, Res_transform)
    resnet50_features=pd.DataFrame(ResNet_features)
    resnet50_features.name="ResNet50"
    
    # %%
    # Point2 clustering
    Clip_kmeans_cluster=Calculate_cluster_Kmeans(CLIP_features)
    Dino_kmeans_cluster=Calculate_cluster_Kmeans(DINOv2_features)
    ResNet_kmeans_cluster=Calculate_cluster_Kmeans(resnet50_features)
    Clip_dbscan_cluster=Calculate_cluster_DBSCAN(CLIP_features)
    Dino_dbscan_cluster=Calculate_cluster_DBSCAN(DINOv2_features)
    ResNet_dbscan_cluster=Calculate_cluster_DBSCAN(resnet50_features)
    
    # %%
    # Point3 Comparison between softlabel and cluster obtained
    dataframes=[CLIP_features,DINOv2_features,resnet50_features]
    CLIP_features.to_pickle(dataFolder+"CLIP.pkl")
    DINOv2_features.to_pickle(dataFolder+"DINOv2.pkl")
    resnet50_features.to_pickle(dataFolder+"ResNet.pkl")
    # %%
    destfile1=folder_name+"cluster_style.json"
    destfile2=folder_name+"cluster_style_sum.json"
    summarize_comparison(SoftLabel_df,dataframes,destfile1,destfile2,prompt)
    # %%
    # Point4 Visualizzation
    # UMAP visualizzation
    '''
    dfolder="./instance_2026-04-24_21-04-16/"+"UMAP_pictures/"
    CLIP_features=pd.read_pickle("D:\progetto_ai\classifier_vehicle\code\instance_2026-04-24_21-04-16\data\CLIP.pkl")
    CLIP_features.name="CLIP"
    DINOv2_features=pd.read_pickle("D:\progetto_ai\classifier_vehicle\code\instance_2026-04-24_21-04-16\data\DINOv2.pkl")
    DINOv2_features.name="DINOv2"
    resnet50_features=pd.read_pickle("D:\progetto_ai\classifier_vehicle\code\instance_2026-04-24_21-04-16\data\ResNet.pkl")
    resnet50_features.name="ResNet50"
    dataframes=[CLIP_features,DINOv2_features,resnet50_features]
    image_dir="dataset/cars_train/cars_train/"
    prototypeFolder="./instance_2026-04-24_21-04-16/"+"prototype/"
    '''
    dfolder=folder_name+"UMAP_pictures/"
    os.makedirs(dfolder, exist_ok=True)
    
    UMAP_visualizzation(CLIP_features,DINOv2_features,resnet50_features,dfolder)
    # %%
    # Prototype extraction and grad-cam
    Methods_clustering=["Kmeans","DBSCAN"]
    prototype_extract(dataframes,image_dir,prototypeFolder,Methods_clustering)
    #print(prototypeFolder)
    
# %%  
if __name__ == "__main__":
    main()


# %%
