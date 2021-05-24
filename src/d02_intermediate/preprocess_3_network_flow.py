# process the network flow data
# three inputs: mobility flow, transaction flow with age bins, and transaction flow with activity bins
# outputs: sa2_edge_flow - network edges with flow and transaction data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# system path
import sys
import os

# util path
utility_path = os.path.join(os.getcwd(),'src/d00_utils/')
sys.path.append(utility_path)
import utilities as util

# data path
# sw: define the path based on the root project directory.
raw_data_path = os.path.join(os.getcwd(),'data/01_raw/')
intermediate_data_path = os.path.join(os.getcwd(),'data/02_intermediate/')

# mount_path = "/Users/shenhaowang/Dropbox (MIT)/project_econ_opportunity_south_Australia"

# read files
flow_df = pd.read_csv(raw_data_path + "flows_sa2_months_2018-02-01.csv")
trans_age_df = pd.read_csv(raw_data_path + "transaction_age_bins.csv")
trans_mcc_df = pd.read_csv(raw_data_path + "transaction_mcc.csv")
sa2_adelaide_edge = gpd.read_file(intermediate_data_path+'shapefiles/sa2_adelaide_edge.shp')

# Change sa2 values to string
flow_df['sa2'] = flow_df.sa2.astype(str)
flow_df['agent_home_sa2'] = [x[:-2] for x in flow_df['agent_home_sa2'].astype(str)]
trans_age_df['source_sa2'] = trans_age_df['source_sa2'].astype(str)
trans_age_df['target_sa2'] = trans_age_df['target_sa2'].astype(str)
trans_mcc_df['source_sa2'] = trans_mcc_df['source_sa2'].astype(str)
trans_mcc_df['target_sa2'] = trans_mcc_df['target_sa2'].astype(str)

# remove invalid values
invalid_value_list = ['Cell Size Limit', 'nan', 'OUTST']
trans_age_df=trans_age_df.loc[~trans_age_df.source_sa2.isin(invalid_value_list)]
trans_age_df=trans_age_df.loc[~trans_age_df.target_sa2.isin(invalid_value_list)]
trans_mcc_df=trans_mcc_df.loc[~trans_mcc_df.source_sa2.isin(invalid_value_list)]
trans_mcc_df=trans_mcc_df.loc[~trans_mcc_df.target_sa2.isin(invalid_value_list)]

# choose only the adelaide area
adelaide_sa4_set = ['401','402','403','404']
flow_adelaide_df = flow_df.loc[np.array([x[:3] in adelaide_sa4_set for x in flow_df.agent_home_sa2])]
flow_adelaide_df = flow_adelaide_df.loc[np.array([x[:3] in adelaide_sa4_set for x in flow_adelaide_df.sa2])]
trans_age_adelaide_df = trans_age_df.loc[np.array([x[:3] in adelaide_sa4_set for x in trans_age_df.source_sa2])]
trans_age_adelaide_df = trans_age_adelaide_df.loc[np.array([x[:3] in adelaide_sa4_set for x in trans_age_adelaide_df.target_sa2])]
trans_mcc_adelaide_df = trans_mcc_df.loc[np.array([x[:3] in adelaide_sa4_set for x in trans_mcc_df.source_sa2])]
trans_mcc_adelaide_df = trans_mcc_adelaide_df.loc[np.array([x[:3] in adelaide_sa4_set for x in trans_mcc_adelaide_df.target_sa2])]

print(flow_adelaide_df.shape)
print(trans_age_adelaide_df.shape)
print(trans_mcc_adelaide_df.shape)

# replace names and reindex
flow_adelaide_df.rename(columns={'agent_home_sa2':'O',
                                 'sa2':'D',
                                 'unique_agents':'flow_agents',
                                 'sum_stay_duration':'flow_duration',
                                 'total_stays':'flow_stays'},inplace=True)
flow_adelaide_df.reset_index(drop=True, inplace=True)
trans_age_adelaide_df.rename(columns={'source_sa2':'O','target_sa2':'D'},inplace=True)
trans_age_adelaide_df.reset_index(drop=True, inplace=True)
trans_mcc_adelaide_df.rename(columns={'source_sa2':'O','target_sa2':'D'},inplace=True)
trans_mcc_adelaide_df.reset_index(drop=True, inplace=True)

# aggregate transaction data into OD pairs
trans_age_adelaide_agg_df=trans_age_adelaide_df.groupby(['O', 'D'],as_index=False).aggregate(["sum"]).reset_index()[['O','D','count','amount']]
trans_mcc_adelaide_agg_df=trans_mcc_adelaide_df.groupby(['O', 'D'],as_index=False).aggregate(["sum"]).reset_index()[['O','D','count','amount']]
trans_age_adelaide_agg_df.columns=['O','D','consumption_count_age_source','consumption_amount_age_source']
trans_mcc_adelaide_agg_df.columns=['O','D','consumption_count_mcc_source','consumption_amount_mcc_source']

# merge files
sa2_adelaide_edge_flow = sa2_adelaide_edge.merge(flow_adelaide_df[['O','D','flow_agents','flow_duration','flow_stays']], on=['O','D'], how='outer')
sa2_adelaide_edge_flow = sa2_adelaide_edge_flow.merge(trans_age_adelaide_agg_df, on=['O','D'], how='outer')
sa2_adelaide_edge_flow = sa2_adelaide_edge_flow.merge(trans_mcc_adelaide_agg_df, on=['O','D'], how='outer')

# save
sa2_adelaide_edge_flow.to_pickle(intermediate_data_path+'sa2_edge_flow.pickle')


















