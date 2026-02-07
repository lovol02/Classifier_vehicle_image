#This file aim to process the metadata and construct
#the propriate dataframe of pandas to let the meta data be structured
import os
import scipy.io
import pandas as pd



def parse_car_label(label,multi_word_brands):

    parts = label.split(' ')
    year = parts[-1]        # Always the last element
    brand=parts[0]+" "+parts[1]
    if brand not in multi_word_brands:
        brand = parts[0]
    #print(brand)
    # Body type is usually the last word before the year 
    # (e.g., "Sedan", "SUV", "Convertible")
    body_type = parts[-2] 
    
    return pd.Series([brand, year, body_type])


def dataframe(meta_file,anno_file,MLB):
    annotations = scipy.io.loadmat(anno_file)['annotations'][0]
    meta = scipy.io.loadmat(meta_file)['class_names'][0]
    
    #extract class name
    class_names=[c[0] for c in meta]
    #construct the structure for annotations
    data=[]
    for ann in annotations:
        data.append({
            "fname": ann['fname'][0],         # filename
            "class_idx": ann['class'][0][0],  # Class index (1-196)
            #boundary box bbox_x1, bbox_y1, bbox_x2, bbox_y2
            "bbox": [ann['bbox_x1'][0][0], ann['bbox_y1'][0][0], ann['bbox_x2'][0][0],ann['bbox_y2'][0][0]]
        })
        
    #each element in data list is a row
    df = pd.DataFrame(data)
    # Map index to actual name (adjusting for 1-based indexing in Matlab)
    df['full_label'] = df['class_idx'].apply(lambda x: class_names[x-1])
    df[['brand', 'year', 'body_type']] = df['full_label'].apply(parse_car_label, args=(MLB,))
    #Now we got 7 column, fname,class_id,bbox,full label, brand, year, body_type
    #contollolato manualmente che corrisponde
    '''
    print(f"ID {196} maps to: {df[df['class_idx'] == 196]['full_label'].iloc[0]}")

    for d in data:
        if d['class_idx']==196:
            print(d['fname'])
    '''
    
    return df


def save_df(df):
    df.to_pickle('data/standard_cars_metadata.pkl')

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)    
    image_dir="dataset/cars_train/cars_train"
    prompt=["sporty aggressive", "elegant luxury", "rugged off-road", "futuristic", "vintage"]
    meta_file = "dataset/car_devkit/devkit/cars_meta.mat"
    anno_file = "dataset/car_devkit/devkit/cars_train_annos.mat"
    train_folder = "dataset/cars_train/images_small"
    multi_word_brands = ["Aston Martin", "Land Rover","Alfa Romeo","General Motors","Tata Motors"]

    df=dataframe(meta_file,anno_file,multi_word_brands)
    save_df(df)
