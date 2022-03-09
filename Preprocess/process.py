import pandas as pd
import numpy as np
from slidewindow import *
from shuffle_data import *
from normalization import *
import pdb

def read_datasets(path):
    df_chunk = pd.read_csv(path, chunksize=1000, header = None)
    
    chunk_list = []  # append each chunk df here 

    # Each chunk is in df format
    for chunk in df_chunk:  
        # perform data filtering 


        # Once the data filtering is done, append the chunk to list
        chunk_list.append(chunk)

    # concat the list into dataframe 
    info = pd.concat(chunk_list)
    return info


def clean_data(data):
	
	data_label = data[["attack_cat"]].copy()
	
	data_trimed = data.iloc[:, : len(data.columns) -1 ].copy() #exclude the 'Label' columns
	
	data_trimed = data_trimed.apply(pd.to_numeric,errors='coerce')
	data_trimed = data_trimed.fillna(data_trimed.mean())
	data_trimed = data_trimed.fillna(0.0)
	data_label.loc[data_label['attack_cat'] == 'BENIGN' , "num_label"] = 0.0 #create new col "num_label" for numeric label
	data_label.loc[data_label['attack_cat'] != 'BENIGN' , "num_label"] = 1.0
	
	data_final = pd.concat([data_label['num_label'], data_trimed], axis = 1)
	data_final = data_final.round(5)
	
	return data_final


if __name__ == '__main__':

    lis=[r'/raid/cs19btech11012/AnomalyDetection/UNSW15/UNSW-NB15-20220127T170817Z-001/UNSW-NB15/ANID/data/UNSW-NB15_1.csv'
         ,r'/raid/cs19btech11012/AnomalyDetection/UNSW15/UNSW-NB15-20220127T170817Z-001/UNSW-NB15/ANID/data/UNSW-NB15_2.csv',
         r'/raid/cs19btech11012/AnomalyDetection/UNSW15/UNSW-NB15-20220127T170817Z-001/UNSW-NB15/ANID/data/UNSW-NB15_3.csv',
         r'/raid/cs19btech11012/AnomalyDetection/UNSW15/UNSW-NB15-20220127T170817Z-001/UNSW-NB15/ANID/data/UNSW-NB15_4.csv']
    df=pd.DataFrame()
    for x in lis:
        df=pd.concat([df,read_datasets(x)])

    df.reset_index(inplace = True)
    df.drop('index',axis=1,inplace=True)

    headers_df=pd.read_csv(r'/raid/cs19btech11012/AnomalyDetection/UNSW15/UNSW-NB15-20220127T170817Z-001/UNSW-NB15/ANID/data/NUSW-NB15_features.csv',encoding='cp1252')
    headers_list=headers_df['Name'].to_list()
    df.columns =headers_list


    df['attack_cat'].unique()
    df['attack_cat'] = df['attack_cat'].fillna('BENIGN')
    df['attack_cat'].unique()

	# selected features from pcc score. refer to features_select_pcc.ipynb for more details 
    features_select = ['num_label', 'sport', 'dsport', 'sttl', 'dttl', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'dmeansz', 'Stime', 'Ltime', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

    allreduced_select_new = clean_data(df)[features_select]
    raw_labels =  df[['attack_cat']].values
    
     #normalize data
    normed_all = store_normed(allreduced_select_new)

    #create sequence data with sliding window
    window_size = 10
    overlap = 9
    slide_all = create_windows(normed_all, window_size, overlap) 
    slide_string_labels = create_windows_raw_label(raw_labels, window_size, overlap) 

    print ('\n shuffling...')
    # shuffle sequence data and create train, val and test datasets
    Shuffle(slide_all).output_data()
    Shuffle(slide_string_labels).output_attack_idx()
