import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle

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
with open(raw_data_path+"OD_Google_API_raw.pickle", 'rb') as w:
    OD_time_raw=pickle.load(w)

with open(raw_data_path+"OD_Google_API_With_Map_Info.pickle", 'rb') as w:
    OD_time_SA2=pickle.load(w)

# rename
OD_time_SA2.rename(columns={'o_sa2_idx':'O',
                    'd_sa2_idx':'D',
                    'od_duration_value':'od_duration',
                    'od_distance_value':'od_distance'}, inplace=True)

# save
OD_time_SA2[['O','D','od_duration','od_distance']].to_pickle(intermediate_data_path+'sa2_edge_travel_time.pickle')




