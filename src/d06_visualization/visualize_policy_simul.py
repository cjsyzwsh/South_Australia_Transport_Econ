# Visualize the change of nodal and link attributes with policy simulation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import normalize
from sklearn import linear_model

# system path
import sys
import os

# util path
utility_path = os.path.join(os.getcwd(),'src/d00_utils/')
sys.path.append(utility_path)
import utilities as util

# path
processing_data_path = os.path.join(os.getcwd(),'data/03_processed/')
model_path = os.path.join(os.getcwd(),'data/04_models/')
model_output_path = os.path.join(os.getcwd(),'data/05_model_outputs/')
report_path = os.path.join(os.getcwd(),'data/06_reporting/')

# read files
with open(processing_data_path+'edge_shp.pickle', 'rb') as f:
    edge_shp = pickle.load(f)

with open(processing_data_path+'node_shp.pickle', 'rb') as f:
    node_shp = pickle.load(f)

with open(model_output_path+'edge_df_policy_simulation.pickle', 'rb') as f:
    edge_df_policy_simulation = pickle.load(f)

with open(model_output_path+'node_df_policy_simulation.pickle', 'rb') as f:
    node_df_policy_simulation = pickle.load(f)


# merge
# five columns are relevant:
# edge_df_policy_simulation: od_duration_save, consumption_amount_increase, consumption_count_increase, flow_agents_increase
# node_df_policy_simulation: income_increase
edge_shp = edge_shp.merge(edge_df_policy_simulation[['O','D','od_duration_save','consumption_amount_increase','consumption_count_increase','flow_agents_increase']],
                          left_on=['O','D'], right_on=['O','D'],how='inner')

node_shp = node_shp.merge(node_df_policy_simulation[['SA2_MAIN16','income_increase','job_based_consumption_opportunity_increase','pop_based_consumption_opportunity_increase',
                                                     'amenity_based_consumption_opportunity_increase','diversity_based_consumption_opportunity_increase']],
                          on=['SA2_MAIN16'],how='inner')

# visualize node change
# sparse = True # control the visual density in the visualization
column_name = 'income_increase'
title_name = 'Increase of median income'
save_path = report_path+'policy_simulation/'
fig_name = 'simulation_income_increase'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path)

# ratio
node_shp['income_increase_ratio'] = node_shp['income_increase']/node_shp['median_income_per_job_aud_persons']
column_name = 'income_increase_ratio'
title_name = 'Increase of median income ratio'
save_path = report_path+'policy_simulation/'
fig_name = 'simulation_income_increase_ratio'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path)

#
column_name = 'pop_based_consumption_opportunity_increase'
title_name = 'Increase of population-based consumption opportunities'
save_path = report_path+'policy_simulation/'
fig_name = 'simulation_pop_based_consumption_opp'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path)

column_name = 'job_based_consumption_opportunity_increase'
title_name = 'Increase of job-based consumption opportunities'
save_path = report_path+'policy_simulation/'
fig_name = 'simulation_job_based_consumption_opp'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path)

column_name = 'amenity_based_consumption_opportunity_increase'
title_name = 'Increase of amenity-based consumption opportunities'
save_path = report_path+'policy_simulation/'
fig_name = 'simulation_amenity_based_consumption_opp'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path)

column_name = 'diversity_based_consumption_opportunity_increase'
title_name = 'Increase of diversity-based consumption opportunities'
save_path = report_path+'policy_simulation/'
fig_name = 'simulation_diversity_based_consumption_opp'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path)





# visualize edge changes
sparse = False # control the visual density in the visualization. Here I think non-sparse visualization is better.
plot_dic = [('od_duration_save', 'Travel time saved', 'simulation_od_duration_save'),
            ('consumption_amount_increase', 'Increase of consumption amount', 'simulation_consumption_amount_increase'),
            ('consumption_count_increase', 'Increase of consumption counts', 'simulation_consumption_count_increase'),
            ('flow_agents_increase', 'Increase of people flow', 'simulation_flow_agents_increase')]

for each_plot in plot_dic:
    column_name = each_plot[0]
    title_name = each_plot[1]
    save_path = report_path+'policy_simulation/'
    if not sparse:
        fig_name = each_plot[2]
        util.plot_sa2_edge_attributes(edge_shp, node_shp, column_name, title_name, fig_name, save_path)
    else:
        fig_name = each_plot[2]+'_sparse'
        edge_shp_sparse = edge_shp.loc[edge_shp[column_name] > np.mean(edge_shp[column_name])+np.std(edge_shp[column_name]), :]
        util.plot_sa2_edge_attributes(edge_shp_sparse, node_shp, column_name, title_name, fig_name, save_path)














