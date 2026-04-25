import pandas as pd

pd.set_option('display.max_columns', None)

def assign_softlabel(df,styles):
    df['soft_label'] = df[styles].idxmax(axis=1)

def merge_softlabel_with_resultDf(softlabelDf,outDf):
    #use inner to select the file which appears in both part
    outDf = pd.merge(outDf, softlabelDf[['file_name', 'soft_label']], on='file_name', how='inner')
    #print(outDf.columns)
    return outDf
if __name__ == "__main__":
    prompt=["sporty aggressive", "elegant luxury", "rugged off-road", "futuristic", "vintage"]
    in_filename='data/standard_cars_softLabel.pkl'
    out_filenames=['data/standard_CLIP_feature.pkl','data/standard_DINOv2_feature.pkl','data/standard_ResNet50_feature.pkl']
    df=pd.read_pickle(in_filename)
    assign_softlabel(df,prompt)
    print(df.head(20))   
    '''
    for f in out_filenames: 
        df1=pd.read_pickle(f)
        
        #df1=df1.drop(['soft_label_x', 'soft_label_y'], axis=1)
        df1=merge_softlabel_with_resultDf(df,df1)
        print(df1.head(5))
        df1.to_pickle(f)
    '''