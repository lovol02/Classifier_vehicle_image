#for DINOv2 model, we have 4 options vits14, vitb14, vitl14, vitg14 
#in order of their dimension(number of parameter),
#Vits14 Fast inference, edge devices, or quick prototyping.
#Vitb14 The "standard" balanced choice for most tasks.
#Vitl14 High accuracy for downstream tasks like depth estimation.
#vitg14 ViT-Giant1.1B1536State-of-the-art performance; requires significant VRAM.
#we use vitb14, 14 stand for patch size is 14x14 pixels, 
#also image size should be multiple of 14
import os
import torch
import torch.hub
from PIL import Image
import torchvision.transforms as T
import pandas as pd

def DINOV2_model_config(device="cuda"):
    if device == "cuda" and torch.cuda.is_available():
        pass
    else:
        device = "cpu"
    # Load the model DINOv2 (ViT-Base)

    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    dinov2_model.eval()

    #This used to transform image into propriate format that dinov2 can process
    transform = T.Compose([
        #resize image to 252pixel, then cut it in center of 224 pixel,Bicubic=4x4 neighbor
        T.Resize(252, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        #convert pixel into numbers
        T.ToTensor(),
        #The mean and std, the values inside is considered as constant value, derive from ImageNet
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return device,dinov2_model,transform

#like CLIP but it process image in pixel instead text
def extract_dino_features(image_path,device,dinov2_model,transform):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        # DINOv2 returns the global embedding
        #should be 768 features
        features = dinov2_model(img_t)
        
        # L2 normalization(we need normalization for clustering and compare)
        features /= features.norm(dim=-1, keepdim=True)
        
    return features.detach().cpu().to(torch.float32).numpy().flatten()


def All_Features_DINOv2(df,dir,device,dinov2_model,transform):
    all_features=[]
    #for index, row in df.head(50).iterrows():
    for index, row in df.iterrows():
        image_path=dir+'/'+row['fname']
        print(image_path)
        vehicle_metadata = {
            "year": row['year'],
            "category": row["body_type"],
            "brand": row["brand"],
            "file_name": row["fname"]
        }
        features=extract_dino_features(image_path,device,dinov2_model,transform)
        #print(features)
        vehicle_metadata['features']=features
        all_features.append(vehicle_metadata)
        #print(all_features)

    return all_features
def init(image_dir,metadata,destFeatureFile,device="cuda"):
    device,dinov2_model,transform=DINOV2_model_config(device)
    df=pd.read_pickle(metadata)
    features=All_Features_DINOv2(df,image_dir,device,dinov2_model,transform)
    DINOv2_features=pd.DataFrame(features)
    DINOv2_features.to_pickle(destFeatureFile)
    
if __name__ == '__main__':
    os.makedirs("data", exist_ok=True) 
    image_dir="dataset/cars_train/cars_train"
    metadata='data/standard_cars_metadata.pkl'
    destFeatureFile='data/standard_DINOv2_feature.pkl'
    init(image_dir,metadata,destFeatureFile)
    
