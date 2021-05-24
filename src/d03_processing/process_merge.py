# This script combines the intermediate data into processed data
# outputs:
#   node.pickle - shape: (110 * _) - all the nodal information
#   edge:pickle - shape: (12100 * _) - all the edge information

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import copy

# system path
import sys
import os

# util path
utility_path = os.path.join(os.getcwd(),'src/d00_utils/')
sys.path.append(utility_path)
import utilities as util

# data path
# sw: define the path based on the root project directory.
intermediate_data_path = os.path.join(os.getcwd(),'data/02_intermediate/')
processing_data_path = os.path.join(os.getcwd(),'data/03_processed/')

# read files
sa2_adelaide_node = gpd.read_file(intermediate_data_path+'shapefiles/sa2_adelaide.shp')
sa2_adelaide_edge = gpd.read_file(intermediate_data_path+'shapefiles/sa2_adelaide_edge.shp')
# read six files
with open(intermediate_data_path+'sa2_node_with_socio_econ_df.pickle', 'rb') as f:
    sa2_node_with_socio_econ_df = pickle.load(f)

with open(intermediate_data_path+'sa2_node_with_POI_counts_entropy.pickle', 'rb') as f:
    sa2_node_with_POI_counts_entropy = pickle.load(f)

with open(intermediate_data_path+'sa2_node_with_only_transport_attributes.pickle', 'rb') as f:
    sa2_node_with_only_transport_attributes = pickle.load(f)

with open(intermediate_data_path+'sa2_edge_flow.pickle', 'rb') as f:
    sa2_edge_flow = pickle.load(f)

with open(intermediate_data_path+'sa2_edge_travel_time.pickle', 'rb') as f:
    sa2_edge_travel_time = pickle.load(f)

with open(intermediate_data_path+'sa2_edge_with_only_transport_attributes.pickle', 'rb') as f:
    sa2_edge_with_only_transport_attributes = pickle.load(f)

# drop geometry columns
sa2_edge_flow.drop(columns=['geometry'],inplace=True)
sa2_edge_with_only_transport_attributes.drop(columns=['geometry'],inplace=True)
sa2_node_with_POI_counts_entropy.drop(columns=['geometry'],inplace=True)
sa2_node_with_only_transport_attributes.drop(columns=['geometry'],inplace=True)

# only subset of socio-econ variables are needed for now.
sa2_node_with_socio_econ_df = sa2_node_with_socio_econ_df[['sa2_code16',
                                                           'num_jobs_000_persons',
                                                           'median_income_per_job_aud_persons',
                                                           'tot_num_jobs_000',
                                                           'total_pop',
                                                           'avg_age',
                                                           'male_percent',
                                                           'female_percent',
                                                           'bachelor_degree_percent',
                                                           'master_degree_percent',
                                                           'perc_indig_age_0_14',
                                                           'perc_indig_age_15_34',
                                                           'perc_indig_age_35_64',
                                                           'poverty_rate_1',
                                                           'poverty_rate_2',
                                                           'unemployment_rate',
                                                           'median_inc',
                                                           'gini']]

#region 1. merge edge files
# sa2_adelaide_edge, sa2_edge_flow, sa2_edge_travel_time, sa2_edge_with_only_transport_attributes
# disaggregate nodal files: sa2_node_with_POI_counts_entropy, sa2_node_with_socio_econ_df.
edge = copy.copy(sa2_adelaide_edge)

# merge with edge files
edge = edge.merge(sa2_edge_flow, left_on=['O','D'], right_on=['O','D'], how='left')
edge = edge.merge(sa2_edge_travel_time, left_on=['O','D'], right_on=['O','D'], how='left')
edge = edge.merge(sa2_edge_with_only_transport_attributes, left_on=['O','D'], right_on=['O','D'], how='left')

# add nodal files to the edge files
edge = edge.merge(sa2_node_with_POI_counts_entropy, left_on=['O'], right_on=['SA2_MAIN16'], how='left')
edge = edge.merge(sa2_node_with_POI_counts_entropy, left_on=['D'], right_on=['SA2_MAIN16'], suffixes=['_o', '_d'], how='left')
edge = edge.merge(sa2_node_with_socio_econ_df, left_on=['O'], right_on=['sa2_code16'], how='left')
edge = edge.merge(sa2_node_with_socio_econ_df, left_on=['D'], right_on=['sa2_code16'], suffixes=['_o', '_d'], how='left')

# save edge
edge.to_pickle(processing_data_path+'edge_shp.pickle')

#endregion


#region 2. merge nodal files
# sa2_adelaide_node, sa2_node_with_socio_econ_df, sa2_node_with_POI_counts_entropy, sa2_node_with_only_transport_attributes;
node = copy.copy(sa2_adelaide_node)
node = node.merge(sa2_node_with_socio_econ_df, left_on=['SA2_MAIN16'], right_on=['sa2_code16'], how='left')
node = node.merge(sa2_node_with_POI_counts_entropy, left_on=['SA2_MAIN16'], right_on=['SA2_MAIN16'], how='left')
node = node.merge(sa2_node_with_only_transport_attributes, left_on=['SA2_MAIN16'], right_on=['SA2_MAIN16'], how='left')

# aggregate edge info to O and D. sa2_edge_flow
# Then add O and D info to the nodal file
sa2_flow_o = sa2_edge_flow.groupby('O').aggregate(['sum']).reset_index()
col_list = sa2_flow_o.columns
col_new = [col[0]+'_o' for col in col_list]
sa2_flow_o.columns = col_new

sa2_flow_d = sa2_edge_flow.groupby('D').aggregate(['sum']).reset_index()
col_list = sa2_flow_d.columns
col_new = [col[0]+'_d' for col in col_list]
sa2_flow_d.columns = col_new

node = node.merge(sa2_flow_o, left_on=['SA2_MAIN16'], right_on=['O_o'], how='left')
node = node.merge(sa2_flow_d, left_on=['SA2_MAIN16'], right_on=['D_d'], how='left')

# drop two columns
node = node.drop(columns=['D_o', 'O_d'])

# change data types
node[['num_nodes', 'num_1degree', 'num_2degree', 'num_3degree', 'num_4degree', 'num_greater5degree']] = \
    node[['num_nodes', 'num_1degree', 'num_2degree', 'num_3degree', 'num_4degree', 'num_greater5degree']].astype('int64')

# add pop density variable
node['pop_density'] = node['total_pop']/(node.area * 1000)

# add POI densities
node['poi_count_density']=node['poi_count']/(1000*node.area)
node['poi_count_agg_density']=node['poi_count_agg']/(1000*node.area)
node['poi_entropy_density']=node['poi_entropy']/(1000*node.area)
node['poi_entropy_agg_density']=node['poi_entropy_agg']/(1000*node.area)

# add job densities
node['job_density_1']=node['num_jobs_000_persons']/(1000*node.area)
node['job_density_2']=node['tot_num_jobs_000']/(1000*node.area)

# add o and d info together
node['flow_agents'] = node['flow_agents_o'] + node['flow_agents_d']
node['flow_duration'] = node['flow_duration_o'] + node['flow_duration_d']
node['flow_stays'] = node['flow_stays_o'] + node['flow_stays_d']
node['consumption_count_age_source'] = node['consumption_count_age_source_o'] + node['consumption_count_age_source_d']
node['consumption_amount_age_source'] = node['consumption_amount_age_source_o'] + node['consumption_amount_age_source_d']
node['consumption_count_mcc_source'] = node['consumption_count_mcc_source_o'] + node['consumption_count_mcc_source_d']
node['consumption_amount_mcc_source'] = node['consumption_amount_mcc_source_o'] + node['consumption_amount_mcc_source_d']

# save node
node.to_pickle(processing_data_path+'node_shp.pickle')
#endregion



#region 3. processing node and edge shapefiles into three data frames
# read node and edge pickles
with open(processing_data_path+'node_shp.pickle', 'rb') as f:
    node = pickle.load(f)

with open(processing_data_path+'edge_shp.pickle', 'rb') as f:
    edge = pickle.load(f)

# create dataframes to facilitate viewing as dataframes in pycharm
node_df = pd.DataFrame(node.drop(columns=['geometry']))
edge_df = pd.DataFrame(edge.drop(columns=['geometry']))

# edit node_df
print(np.sum(node_df.isna()))
print(np.sum(node_df==0))
print(node_df.describe())
# processing na and zeros for regression on the log space
node_df.dropna(how='any',inplace=True)
print(node_df.shape) # dropped 6 observations. Left with 104 obs.
non_zero_masks = np.logical_and(node_df.bachelor_degree_percent>0.00001, node_df.flow_agents_o>0.00001)
non_zero_masks = np.logical_and(node_df.poi_entropy_agg>0.00001, non_zero_masks)
node_df = node_df.loc[non_zero_masks, :]
print(node_df.shape) # dropped 2 observations. Left with 102 obs.

# lift all road attributes by one for the log transformation.
road_attributes = ['class_ART',
                   'class_BUS', 'class_COLL', 'class_HWY', 'class_LOCL',
                   'class_SUBA', 'class_TRK2', 'class_TRK4', 'class_UND', 'num_roads', 'num_nodes',
                   'num_1degree', 'num_2degree', 'num_3degree', 'num_4degree',
                   'num_greater5degree']
node_df[road_attributes] += 1.0


# edit edge_df
print(np.sum(edge_df.isna()))
print(np.sum(edge_df==0))
print(edge_df.describe())
# drop self loops (110) and zero population areas (440)
non_zero_masks = np.logical_and(edge_df.od_duration > 0.000001, edge_df.total_pop_o > 0.000001) # od_duration and total_pop_o
non_zero_masks = np.logical_and(non_zero_masks, edge_df.total_pop_d > 0.000001)
non_zero_masks = np.logical_and(non_zero_masks, edge_df.avg_age_o > 0.000001)
non_zero_masks = np.logical_and(non_zero_masks, edge_df.avg_age_d > 0.000001)
non_zero_masks = np.logical_and(non_zero_masks, edge_df.bachelor_degree_percent_o > 0.000001)
non_zero_masks = np.logical_and(non_zero_masks, edge_df.bachelor_degree_percent_d > 0.000001)
non_zero_masks = np.logical_and(non_zero_masks, edge_df.poi_entropy_agg_o > 0.000001)
non_zero_masks = np.logical_and(non_zero_masks, edge_df.poi_entropy_agg_d > 0.000001)
edge_df = edge_df.loc[non_zero_masks, :]


# attach POI density (O and D) to edge_df
edge_df = edge_df.merge(node_df[['SA2_MAIN16', 'pop_density', 'poi_count_density', 'poi_count_agg_density', 'poi_entropy_density', 'poi_entropy_agg_density', 'job_density_1', 'job_density_2']],
                        left_on = 'O', right_on = 'SA2_MAIN16')
edge_df = edge_df.merge(node_df[['SA2_MAIN16', 'pop_density', 'poi_count_density', 'poi_count_agg_density', 'poi_entropy_density', 'poi_entropy_agg_density', 'job_density_1', 'job_density_2']],
                        left_on = 'D', right_on = 'SA2_MAIN16', suffixes = ['_o', '_d'])


# split edge_df into edge_consumption_df and edge_flow_df
# it is because the consumption data have a lot of missing info.
edge_flow_df = edge_df.loc[~edge_df.flow_agents.isna(), ]
edge_consumption_df = edge_df.loc[~edge_df.consumption_count_age_source.isna(), ]

#
edge_flow_df = edge_flow_df.drop(columns = ['consumption_count_age_source', 'consumption_amount_age_source',
                                            'consumption_count_mcc_source', 'consumption_amount_mcc_source'])
edge_flow_df.dropna(how='any', inplace=True)
edge_consumption_df.dropna(how='any', inplace=True)

# lift road attributes by one unit.
edge_flow_df[road_attributes] += 1.0
edge_consumption_df[road_attributes] += 1.0

# print dimensions
print("Final edge mobility flow dataframe dim is: ", edge_flow_df.shape)
print("Final edge consumption flow dataframe dim is: ", edge_consumption_df.shape)
print("Final node dataframe dim is: ", node_df.shape)

# check the final
print(np.sum(node_df.isna()))
print(np.sum(node_df == 0))
print(np.sum(edge_flow_df.isna()))
print(np.sum(edge_flow_df == 0))
print(np.sum(edge_consumption_df.isna()))
print(np.sum(edge_consumption_df == 0))

# save dataframes
node_df.to_pickle(processing_data_path+'node_df.pickle')
edge_flow_df.to_pickle(processing_data_path+'edge_flow_df.pickle')
edge_consumption_df.to_pickle(processing_data_path+'edge_consumption_df.pickle')
#endregion







