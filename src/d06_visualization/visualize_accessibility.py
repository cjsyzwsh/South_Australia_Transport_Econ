# visualize the status quo accessibility metrics.

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

#
node_shp = node_shp.merge(node_df_policy_simulation[['SA2_MAIN16','pop_based_consumption_opportunity','job_based_consumption_opportunity',
                                                     'amenity_based_consumption_opportunity','diversity_based_consumption_opportunity']],
                          on=['SA2_MAIN16'],how='inner')

# reload node_shp for the visualization function
with open(processing_data_path+'node_shp.pickle', 'rb') as f:
    node_shp_complete = pickle.load(f)


# plot the accessibility metrics
column_name = 'pop_based_consumption_opportunity'
title_name = 'Population-based consumption opportunities'
save_path = report_path+'accessibility_metrics/'
fig_name = 'pop_based_consumption_opp'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path, node_shp_complete)

column_name = 'job_based_consumption_opportunity'
title_name = 'Job-based consumption opportunities'
save_path = report_path+'accessibility_metrics/'
fig_name = 'job_based_consumption_opp'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path, node_shp_complete)

column_name = 'amenity_based_consumption_opportunity'
title_name = 'Amenity-based consumption opportunities'
save_path = report_path+'accessibility_metrics/'
fig_name = 'amenity_based_consumption_opp'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path, node_shp_complete)

column_name = 'diversity_based_consumption_opportunity'
title_name = 'Diversity-based consumption opportunities'
save_path = report_path+'accessibility_metrics/'
fig_name = 'diversity_based_consumption_opp'
util.plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path, node_shp_complete)






















