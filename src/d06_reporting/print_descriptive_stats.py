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

# util path
utility_path = os.path.join(os.getcwd(),'src/d00_utils/')
sys.path.append(utility_path)
import utilities as util

# data path
# sw: define the path based on the root project directory.
intermediate_data_path = os.path.join(os.getcwd(),'data/02_intermediate/')
processing_data_path = os.path.join(os.getcwd(),'data/03_processed/')

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

### printing
print("Number of flow observations is: ", edge_flow_df[['flow_agents']].shape[0])
print("Number of consumption counts is: ", edge_consumption_df[['consumption_count_mcc_source']].shape[0])
print("Number of consumption amount is: ", edge_consumption_df[['consumption_amount_mcc_source']].shape[0])
print("Average travel time is: ", edge_flow_df['od_duration'].mean()/60)
print("Average travel distance is: ", edge_flow_df['od_distance'].mean())









































