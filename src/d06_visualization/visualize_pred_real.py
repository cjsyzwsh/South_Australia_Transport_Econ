# evaluate the models by plotting real vs. predicted values

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


# read data
with open(processing_data_path+'node_df.pickle', 'rb') as f:
    node_df = pickle.load(f)

with open(processing_data_path+'edge_consumption_df.pickle', 'rb') as f:
    edge_consumption_df = pickle.load(f)

with open(processing_data_path+'edge_flow_df.pickle', 'rb') as f:
    edge_flow_df = pickle.load(f) # more observations


###########################################################################################
# Visualize predicted vs. real outputs
###########################################################################################
# format: saved model name: (picture title, figure name, dataframe).
model_list_to_read_dic = {'model1_list_od_duration': ('Travel time (origin to destination)', 'pred_travel_time', edge_flow_df),
                          'model2_list_consumption_amount_mcc_source': ('Amount of consumption', 'pred_consumption_amount', edge_consumption_df),
                          'model2_list_consumption_count_mcc_source': ('Counts of consumption', 'pred_consumption_count', edge_consumption_df),
                          'model2_list_flow_agents': ('Flow of people', 'pred_flow', edge_flow_df),
                          'model3_list_median_income_per_job_aud_persons': ('Median income', 'pred_inc', node_df)
                          }

for model_pickle_name in model_list_to_read_dic.keys():
    with open(model_path + model_pickle_name + '.pickle', 'rb') as f:
        model_list = pickle.load(f)

    # get picture title, fig_name, and dataframe
    picture_title, fig_name, df = model_list_to_read_dic[model_pickle_name]

    # obtain y_name, x_name, and saved model
    _, (y_name, x_name), model = model_list[-1]

    # plot the observed vs. predicted values.
    util.plot_observed_predicted(df, y_name, x_name, model, report_path+'model_visual_pred_actual/', picture_title, fig_name)


