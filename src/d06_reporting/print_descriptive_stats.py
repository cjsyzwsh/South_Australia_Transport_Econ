# this script prints out the descriptive stats to facilitate paper writing.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import copy

# system path
import sys
import os
import seaborn as sns

# use ggplot style
plt.style.use('ggplot')

# util path
utility_path = os.path.join(os.getcwd(),'src/d00_utils/')
sys.path.append(utility_path)
import utilities as util

# data path
# sw: define the path based on the root project directory.
intermediate_data_path = os.path.join(os.getcwd(),'data/02_intermediate/')
processing_data_path = os.path.join(os.getcwd(),'data/03_processed/')
model_output_path = os.path.join(os.getcwd(),'data/05_model_outputs/')
report_path = os.path.join(os.getcwd(),'data/06_reporting/')


#
with open(processing_data_path+'node_df.pickle', 'rb') as f:
    node_df = pickle.load(f)

with open(processing_data_path+'edge_consumption_df.pickle', 'rb') as f:
    edge_consumption_df = pickle.load(f)

with open(processing_data_path+'edge_flow_df.pickle', 'rb') as f:
    edge_flow_df = pickle.load(f)

with open(processing_data_path+'edge_shp.pickle', 'rb') as f:
    edge_shp = pickle.load(f)

with open(processing_data_path+'node_shp.pickle', 'rb') as f:
    node_shp = pickle.load(f)

with open(model_output_path+'edge_df_policy_simulation.pickle', 'rb') as f:
    edge_df_policy_simulation = pickle.load(f)

with open(model_output_path+'node_df_policy_simulation.pickle', 'rb') as f:
    node_df_policy_simulation = pickle.load(f)


# printing
print("Number of flow observations is: ", edge_flow_df[['flow_agents']].shape[0])
print("Number of consumption counts is: ", edge_consumption_df[['consumption_count_mcc_source']].shape[0])
print("Number of consumption amount is: ", edge_consumption_df[['consumption_amount_mcc_source']].shape[0])
print("Average travel time is: ", edge_flow_df['od_duration'].mean()/60)
print("Average travel distance is: ", edge_flow_df['od_distance'].mean())

# descriptive tables
# socio econ
descriptive_table = node_df[['median_income_per_job_aud_persons', 'total_pop']].describe().T
descriptive_table.index = ['inc', 'total_pop']
descriptive_table = np.round(descriptive_table, decimals = 0)
with open(report_path+'descriptive_tables/'+ 'socioecon.txt', 'w') as f:
    f.writelines(descriptive_table.to_latex())

# visual
fig,ax = plt.subplots(1,2, figsize = (10, 3))
ax[0].hist(node_df['median_income_per_job_aud_persons'], bins=15, color = 'lightcoral')
ax[1].hist(node_df['total_pop'], bins=15, color = 'bisque')
ax[0].set_title("Median income")
ax[1].set_title("Total population")
fig.savefig(report_path+'descriptive_visual/'+'socioecon.png')
plt.close()


# mobility and consumption flow
descriptive_table = edge_flow_df[['flow_agents']].describe().T
descriptive_table = np.round(descriptive_table, decimals = 0)
with open(report_path+'descriptive_tables/'+ 'mobility_flow.txt', 'w') as f:
    f.writelines(descriptive_table.to_latex())

descriptive_table = edge_consumption_df[['consumption_count_mcc_source', 'consumption_amount_mcc_source']].describe().T
descriptive_table.index = ['consumption counts', 'consumption amount']
descriptive_table = np.round(descriptive_table, decimals = 0)
with open(report_path+'descriptive_tables/'+ 'consumption_flow.txt', 'w') as f:
    f.writelines(descriptive_table.to_latex())

fig,ax = plt.subplots(1,3, figsize = (10, 3))
ax[0].hist(edge_flow_df['flow_agents'], bins=15, color = 'lightcoral')
ax[1].hist(edge_consumption_df['consumption_count_mcc_source'], bins=15, color = 'bisque')
ax[2].hist(edge_consumption_df['consumption_amount_mcc_source'], bins=15, color = 'moccasin')
ax[0].set_title("Mobility flow")
ax[1].set_title("Consumption counts")
ax[2].set_title("Consumption amount")
fig.savefig(report_path+'descriptive_visual/'+'flow.png')
plt.close()


# travel time and distance
descriptive_table = edge_shp[['od_duration', 'od_distance']].describe().T
descriptive_table.index = ['OD Duration', 'OD Distance']
descriptive_table = np.round(descriptive_table, decimals = 0)
with open(report_path+'descriptive_tables/'+ 'travel_time_distance.txt', 'w') as f:
    f.writelines(descriptive_table.to_latex())

fig,ax = plt.subplots(1,2, figsize = (10, 3))
ax[0].hist(edge_shp['od_duration'], bins=15, color = 'lightcoral')
ax[1].hist(edge_shp['od_distance'], bins=15, color = 'bisque')
ax[0].set_title("Travel time")
ax[1].set_title("Travel distance")
fig.savefig(report_path+'descriptive_visual/'+'travel.png')
plt.close()


#
print("Average travel time save ratio is:", np.mean(edge_df_policy_simulation['od_duration_save']/edge_df_policy_simulation['od_duration']))
print("Max travel time save ratio is:", np.max(edge_df_policy_simulation['od_duration_save']/edge_df_policy_simulation['od_duration']))
print("Average income increase ratio is:", np.mean(node_df_policy_simulation['income_increase']/node_df_policy_simulation['median_income_per_job_aud_persons']))
print("Max income increase ratio is:", np.max(node_df_policy_simulation['income_increase']/node_df_policy_simulation['median_income_per_job_aud_persons']))
print("Average econ opportunity increase ratio is:", np.mean(node_df_policy_simulation['amenity_based_consumption_opportunity_increase']/node_df_policy_simulation['amenity_based_consumption_opportunity']))
print("Max econ opportunity increase ratio is:", np.max(node_df_policy_simulation['amenity_based_consumption_opportunity_increase']/node_df_policy_simulation['amenity_based_consumption_opportunity']))
print("Status quo Gini index is: ", util.gini(list(node_df_policy_simulation['median_income_per_job_aud_persons'])))
new_inc = node_df_policy_simulation['median_income_per_job_aud_persons']+node_df_policy_simulation['income_increase']
new_inc.dropna(inplace = True)
print("New Gini index is: ", util.gini(list(new_inc)))








































