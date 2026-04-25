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
#print(features.columns)
#print(len(features))
#print(len(features_DINOv2))

styles=["sporty aggressive", "elegant luxury", "rugged off-road", "futuristic", "vintage"]
softlabel=pd.read_pickle('data/standard_cars_softLabel.pkl')
#use df['soft_label'] =softlabel.set_index('file_name')[styles].idxmax(axis=1) to let every 
#softlabel['soft_label'] = softlabel[styles].idxmax(axis=1)
print(softlabel.head(3))
#map solve the index mismatching of two dataframe
#mapping = softlabel.set_index('file_name')[styles].idxmax(axis=1)
#features['soft_label'] = features['file_name'].map(mapping)
#print(features.head(3))

#print(features.index)
# Likely output: RangeIndex(start=0, stop=100, step=1)

result_of_calculation = softlabel.set_index('file_name')[styles].idxmax(axis=1)
#print(result_of_calculation.index)