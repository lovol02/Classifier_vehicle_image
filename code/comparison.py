#This file aim to compare the zero-shot labe obatained by the CLIP(as a supervised approach) 
#with the unsupervised approach used in our cases, RestNetv50, CLIP, DINOV2
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,homogeneity_score, completeness_score
import json
pd.set_option('display.max_columns', None)

def comparison_kmeans(mergedDf, Kcount,prompt):
    count=0
    kmeans_dic={}
    kmeans_dic_high={}
    while count < Kcount:
        kmeans_dic_high[count]=[]
        result = mergedDf[mergedDf['clusters_kmeans'] == count]
        prob_dic = (result[prompt].sum() / len(result)).to_dict()
        #print(prob_dic)
        for key, elem in prob_dic.items():
            if elem >= 0.30:
                kmeans_dic_high[count].append({key:elem})
                print(f"for kmeans cluster of index {count}  has {len(result)} elements, and this cluster is {elem:.2f}% has style {key}")
        count+=1
        kmeans_dic[count]=prob_dic
    return kmeans_dic,kmeans_dic_high
    
        
def comparison_dbscan(mergedDf, DBcount,prompt):
    count=0
    dbscan_dic={}
    dbscan_dic_high={}
    while count < DBcount:
        dbscan_dic_high[count]=[]
        result = mergedDf[mergedDf['clusters_dbscan'] == count]
        prob_dic = (result[prompt].sum() / len(result)).to_dict()
        #print(prob_dic)
        for key, elem in prob_dic.items():
            if elem >= 0.30:
                dbscan_dic_high[count].append({key:elem})
                print(f"for dbscan cluster of index {count}  has {len(result)} elements, and this cluster is {elem:.2f}% has style {key}")
        count+=1
        dbscan_dic[count]=prob_dic
    return dbscan_dic,dbscan_dic_high
        
        
def comparison_Softlabel_Cluster(softlabelDf,clusteredDf,prompt):
    
    cols_to_select= ['file_name'] + prompt
    kmeans_count=len(clusteredDf['clusters_kmeans'].value_counts())
    dbscan_count=len(clusteredDf['clusters_dbscan'].value_counts())
    mergedDf= pd.merge(clusteredDf, softlabelDf[cols_to_select], on='file_name')
    kmeans_dic,kmeans_style_dic= comparison_kmeans(mergedDf, kmeans_count,prompt)
    print("-----------------------------------------\n")
    # The count in dbscan should -1 than the value get from df, 
    # cause there is possible unclusterd value in cluster column
    dbscan_dic,dbscan_style_dic= comparison_dbscan(mergedDf, dbscan_count-1,prompt)
    return kmeans_dic,dbscan_dic,kmeans_style_dic,dbscan_style_dic

def save_report(file,dic):
    with open(file, 'w') as f:
        json.dump(dic, f, indent=4)

def summarize_comparison(softlabelDF,Dataframes,destfile1,destfile2,prompt):
    dic_sum={}
    dic_style_sum={}
    
    for df in Dataframes:
        kmeans_dic,dbscan_dic,kmeans_style_dic,dbscan_style_dic=comparison_Softlabel_Cluster(softlabelDF,df,prompt)
        dic_sum[df.name+"_kmeans"]=kmeans_dic
        dic_sum[df.name+"_dbscan"]=dbscan_dic
        dic_style_sum[df.name+"_kmeans"]=kmeans_style_dic
        dic_style_sum[df.name+"_dbscan"]=dbscan_style_dic
        print("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000\n")
    save_report(destfile1,dic_sum)
    save_report(destfile2,dic_style_sum)

if __name__ == "__main__":
    prompt=["sporty aggressive", "elegant luxury", "rugged off-road", "futuristic", "vintage"]
    filenames=['data/standard_CLIP_feature.pkl','data/standard_DINOv2_feature.pkl','data/standard_ResNet50_feature.pkl']
    df=pd.read_pickle('data/standard_cars_softLabel.pkl')
    #df1=pd.read_pickle('data/standard_CLIP_feature.pkl')
    #print(len(df1['clusters_kmeans'].value_counts()))
    #comparison(df,df1,prompt)
    location1="temp/cluster_style.json"
    location2="temp/cluster_style_sum.json"

    dfs=[]
    for f in filenames:
        df1=pd.read_pickle(f)
        df1.name=f.split(".")[0]
        dfs.append(df1)
    summarize_comparison(df,dfs,location1,location2,prompt)