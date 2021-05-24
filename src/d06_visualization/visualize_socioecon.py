# visualization
# type 1. Visualize the nodes' attributes (sociodemographics, etc.)
# type 2. Visualize the edges' attributes (flows, etc.)
# type 3. Visualize the edges' attributes in a sparse manner (flows, etc.)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import statsmodels.api as sm
import copy

# system path
import sys
import os

# util path
utility_path = os.path.join(os.getcwd(),'src/d00_utils/')
sys.path.append(utility_path)
import utilities as util

# path
processing_data_path = os.path.join(os.getcwd(),'data/03_processed/')
report_path = os.path.join(os.getcwd(),'data/06_reporting/')

# read data
with open(processing_data_path+'node_df.pickle', 'rb') as f:
    node_df = pickle.load(f)

with open(processing_data_path+'edge_consumption_df.pickle', 'rb') as f:
    edge_consumption_df = pickle.load(f)

with open(processing_data_path+'edge_flow_df.pickle', 'rb') as f:
    edge_flow_df = pickle.load(f) # more observations

with open(processing_data_path+'node_shp.pickle', 'rb') as f:
    node_shp = pickle.load(f)

with open(processing_data_path+'edge_shp.pickle', 'rb') as f:
    edge_shp = pickle.load(f)

# visualize node_shp
plot_dic = [('total_pop', 'Total population', 'node_socioecon_total_pop'),
            ('pop_density', 'Population density', 'node_socioecon_pop_density'),
            ('tot_num_jobs_000', 'Total jobs', 'node_socioecon_total_jobs'),
            ('median_income_per_job_aud_persons', 'Median income', 'node_socioecon_median_income'),
            ('master_degree_percent', 'Percent of master degree', 'node_socioecon_percent_master'),
            ('poi_count', 'Counts of POIs', 'node_poi_count'),
            ('poi_count_agg', 'Counts of aggregated POIs', 'node_poi_count_agg'),
            ('poi_entropy', 'Entropy of POIs', 'node_poi_entropy'),
            ('poi_entropy_agg', 'Entropy of aggregated POIs', 'node_poi_entropy_agg'),
            ('poi_count_density', 'Count density of POIs', 'node_poi_count_density'),
            ('poi_count_agg_density', 'Count density of aggregated POIs', 'node_poi_count_agg_density'),
            ('poi_entropy_density', 'Entropy density of POIs', 'node_poi_entropy_density'),
            ('poi_entropy_agg_density', 'Entropy density of aggregated POIs', 'node_poi_entropy_agg_density'),
            ('class_HWY', 'Number of highways', 'node_road_class_HWY'),
            ('num_nodes', 'Number of road intersections', 'node_road_num_nodes'),
            ('num_4degree', 'Number of 4-way intersections', 'node_road_num_4degree'),
            ('flow_agents_o', 'Agent flow (origin aggregated)', 'node_flow_agents_o'),
            ('flow_agents_d', 'Agent flow (destination aggregated)', 'node_flow_agents_d'),
            ('consumption_amount_age_source_o', 'Consumption flow (origin aggregated)', 'node_flow_consumption_amount_age_source_o'),
            ('consumption_amount_age_source_d', 'Consumption flow (destination aggregated)', 'node_flow_consumption_amount_age_source_d')]

for each_plot in plot_dic:
    column_name = each_plot[0]
    title_name = each_plot[1]
    fig_name = each_plot[2]
    save_path = report_path+'node_visual/'
    util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path)

# visualize edge_shp
plot_dic = [('flow_agents', 'Flow of unique agents', 'edge_flow_agents'),
            ('flow_duration', 'Flow of agents duration', 'edge_flow_duration'),
            ('flow_stays', 'Flow of agents stay', 'edge_flow_stays'),
            ('consumption_count_age_source', 'Consumption counts (age source)', 'edge_consumption_count_age_source'),
            ('consumption_amount_age_source', 'Consumption amount (age source)', 'edge_consumption_amount_age_source'),
            ('consumption_count_mcc_source', 'Consumption counts (mcc source)', 'edge_consumption_count_mcc_source'),
            ('consumption_amount_mcc_source', 'Consumption amount (mcc source)', 'edge_consumption_amount_mcc_source'),
            ('od_duration', 'OD duration', 'edge_od_duration'),
            ('class_HWY', 'Highway connectivity', 'edge_class_HWY'),
            ('class_LOCL', 'Local way connectivity', 'edge_class_LOCL'),
            ('num_nodes', 'Number of nodes', 'edge_num_nodes'),
            ('num_4degree', 'Number of 4-way intersections', 'edge_num_4degree')]

# drop the self loops
edge_shp = edge_shp.loc[edge_shp['O'] != edge_shp['D'], :]

for each_plot in plot_dic:
    column_name = each_plot[0]
    title_name = each_plot[1]
    fig_name = each_plot[2]
    save_path = report_path+'edge_visual/'
    util.plot_sa2_edge_attributes(edge_shp, node_shp, column_name, title_name, fig_name, save_path)

# visualize sparse edge plots for the three flow attributes
# plot_dic = [('flow_agents', 'Flow of unique agents', 'edge_flow_agents_sparse'),
#             ('flow_duration', 'Flow of agents duration', 'edge_flow_duration_sparse'),
#             ('flow_stays', 'Flow of agents stay', 'edge_flow_stays_sparse')]

# visualize sparse
for each_plot in plot_dic:
    column_name = each_plot[0]
    title_name = each_plot[1]
    fig_name = each_plot[2]+'_sparse'
    save_path = report_path+'edge_visual/'
    edge_shp_sparse = edge_shp.loc[edge_shp[column_name] > np.mean(edge_shp[column_name])+np.std(edge_shp[column_name]), :]
    util.plot_sa2_edge_attributes(edge_shp_sparse, node_shp, column_name, title_name, fig_name, save_path)


















