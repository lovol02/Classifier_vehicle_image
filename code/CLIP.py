#The CLIP part of this work
import os
import pandas as pd
import torch
import open_clip
from PIL import Image

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)

def CLIP_model_config(device="cuda"):
    if device == "cuda" and torch.cuda.is_available():
        pass
    else:
        device = "cpu"
    #define device use cuda
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #define the open_clip with model ViT-B-32 and dataset laion2b_s34b_b79k
    #where ViT-B-32: Vision Transformer, Base size, with 32x32 pixel patches.
    #laion2b: Trained on the LAION-2B dataset (2 billion English image-text pairs).
    #s34b: The model "saw" 34 billion samples during training (meaning it went through the dataset multiple times).
    #b79k: A massive global batch size of 79,000 was used during training.
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval() #Added to tell model, we are predicting not training.
    model.to(device)
    return device,model,preprocess,tokenizer

def generate_soft_labels(image_path, stylistic_prompts,device,model,preprocess,tokenizer):
    """
    Generates soft labels using CLIP .
    """
    # Load and preprocess image.(Resizes, crops, and normalizes the image so it matches the format CLIP expects)
    # Adds a "batch dimension." Even for one image, PyTorch expects a shape like [1, 3, 224, 224]
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Tokenize prompts
    #Converts your list of strings into numerical tensors (tokens) that the CLIP text encoder can understand.
    text_tokens = tokenizer(stylistic_prompts).to(device)
    #With no_grad disable the gradient calculation, cause we just want predict not train it.
    #With torch.amp.autocast('cuda'), Uses "Mixed Precision" to speed up computation and reduce memory usage on GPUs.
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        # Encode image and prompt features, should be 512 features
        image_features = model.encode_image(image)
        #print(image_features)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features for Cosine Similarity
        #L2 Normalization
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate Cosine Similarity (Soft Labels)
        # Using a temperature scale (100.0) to sharpen the distribution, 
        # without this the probabilities would look very flat and "unsure."
        # image_features @ text_features.T dot production for cosine similarity
        # Then a softmaxt applicate that let sum up of probabilities become 1
        logit_scale = 100.0 
        text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
        
    # Convert to dictionary for easy DataFrame integration
    results = {prompt: prob.item() for prompt, prob in zip(stylistic_prompts, text_probs[0])}
    # Integrate Metadata
    
    #detach() -> detach from vector model(release memories)
    #cpu()-> move to cpu instead gpu
    #to()-> autocast uses float16, but the algorithms in cpu prefer the float32
    #numpy() -> Transform the tensor of PyTorch in an array NumPy
    #flatten() -> Transform the matrix 1x512 into 512 elements
    features_for_cluster = image_features.detach().cpu().to(torch.float32).numpy().flatten()
    #print(features_for_cluster)
    return results,features_for_cluster


def All_SoftLabel_and_Features_CLIP(df,prompt,dir,device,model,preprocess,tokenizer):
    all_results=[]
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
        result,features=generate_soft_labels(image_path,prompt,device,model,preprocess,tokenizer)
        result.update(vehicle_metadata)
        vehicle_metadata['features']=features
        all_features.append(vehicle_metadata)
        #print(all_features)
        all_results.append(result)
        #print(all_result[-1])

    return all_results,all_features

def init(image_dir,prompt,metadata,destSoftLabelFile,destCLIPFeatureFile,device="cuda"):
    device,model,preprocess,tokenizer = CLIP_model_config(device)
    df=pd.read_pickle(metadata)
    softLabel,features=All_SoftLabel_and_Features_CLIP(df,prompt,image_dir,device,model,preprocess,tokenizer)
    SoftLabel_df = pd.DataFrame(softLabel)
    CLIP_features=pd.DataFrame(features)
    SoftLabel_df.to_pickle(destSoftLabelFile)
    CLIP_features.to_pickle(destCLIPFeatureFile)
    
if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)    
    image_dir="dataset/cars_train/cars_train"
    metadata='data/standard_cars_metadata.pkl'
    prompt=["sporty aggressive", "elegant luxury", "rugged off-road", "futuristic", "vintage"]
    destSoftLabelFile='data/standard_cars_softLabel.pkl'
    destCLIPFeatureFile='data/standard_CLIP_feature.pkl'
    init(image_dir,prompt,metadata,destSoftLabelFile,destCLIPFeatureFile)