#ResNet a classical vision based on the CNN(other two is looking on transfomer) 
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T
import pandas as pd

# Load the model ResNet50 with standard weight of ImageNet
weights = models.ResNet50_Weights.IMAGENET1K_V2 #Bicubic interpolation, more accurate
resnet50 = models.resnet50(weights=weights)
#cut off the final layer to get only embedding, without getting a category prediction
#the model stops at the Global Average Pooling layer, tensor has shape [batch_size, 2048, 1, 1]
resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet50 = resnet50.to(device)
resnet50.eval()

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

def extract_resnet50_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        #should be 2048 features, but as last layer is been cut, so the distribution is spartial 
        # (we still need use flatten to get a vector of 2048 elements)
        features = resnet50(img_t)
        #use flatten to reduce the dimension of features to 1D
        features = features.flatten(1)
        # L2 normalization(we need normalization for clustering and compare)
        features /= features.norm(dim=-1, keepdim=True)
        
    return features.detach().cpu().to(torch.float32).numpy().flatten()


def All_Features_ResNet50(df,dir):
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
        features=extract_resnet50_features(image_path)
        print(features)
        vehicle_metadata['features']=features
        all_features.append(vehicle_metadata)
        #print(all_features)

    return all_features


os.makedirs("data", exist_ok=True) 
image_dir="dataset/cars_train/cars_train"
df=pd.read_pickle('data/standard_cars_metadata.pkl')
features=All_Features_ResNet50(df,image_dir)
resnet50_features=pd.DataFrame(features)
resnet50_features.to_pickle('data/standard_ResNet50_feature.pkl')