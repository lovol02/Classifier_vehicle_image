import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

df=pd.read_pickle('data/standard_cars_softLabel.pkl')
#print(df)

features=pd.read_pickle('data/standard_CLIP_feature.pkl')
#print(features)
features_DINOv2=pd.read_pickle('data/standard_DINOv2_feature.pkl')
#print(features.head(1)['features'])

features_ResNet50=pd.read_pickle('data/standard_ResNet50_feature.pkl')
#print(features_ResNet50.head(1)['features'].str.len())
print(features.head(1))
#print(len(features))
#print(len(features_DINOv2))

