# sw: script used to scrape travel time data from Google API.
# No need to run the script for replication.
# TBD: need to set up the global environment variables.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot
from pysal.lib import weights
import networkx as nx
from scipy.spatial import distance
import googlemaps

# system path
import sys
import os

# util path
utility_path = os.path.join(os.getcwd(),'src/d00_utils/')
sys.path.append(utility_path)
import utilities as util

# data path
raw_data_path = os.path.join(os.getcwd(),'data/01_raw/')
intermediate_data_path = os.path.join(os.getcwd(),'data/02_intermediate/')

# read files
sa2_adelaide = gpd.read_file(intermediate_data_path + 'shapefiles/sa2_adelaide.shp')

sa2_adelaide['centroids'] = sa2_adelaide.centroid
sa2_adelaide['Lat'] = sa2_adelaide.centroids.y
sa2_adelaide['Long'] = sa2_adelaide.centroids.x

#
# create a new dataframe
OD = {}
OD['o_idx'] = []
OD['d_idx'] = []
OD['o_sa2_idx'] = []
OD['d_sa2_idx'] = []
OD['o_lat'] = []
OD['o_long'] = []
OD['d_lat'] = []
OD['d_long'] = []

for i in range(sa2_adelaide.shape[0]):
    print("Origin Index is: ", i)
    o_idx = i
    o_sa2_idx = sa2_adelaide.loc[i, 'SA2_MAIN16']
    o_lat = sa2_adelaide.loc[i, 'Lat']
    o_long = sa2_adelaide.loc[i, 'Long']

    for j in range(sa2_adelaide.shape[0]):
        d_idx = j
        d_sa2_idx = sa2_adelaide.loc[j, 'SA2_MAIN16']
        d_lat = sa2_adelaide.loc[j, 'Lat']
        d_long = sa2_adelaide.loc[j, 'Long']

        # append
        OD['o_idx'].append(o_idx)
        OD['d_idx'].append(d_idx)
        OD['o_sa2_idx'].append(o_sa2_idx)
        OD['d_sa2_idx'].append(d_sa2_idx)
        OD['o_lat'].append(o_lat)
        OD['o_long'].append(o_long)
        OD['d_lat'].append(d_lat)
        OD['d_long'].append(d_long)

# create the data frame
OD_df = pd.DataFrame(OD)

# Need to specify your API_key
gmaps = googlemaps.Client(key=API_key)

OD_time_dic = {}

for idx in range(OD_df.shape[0]):
    # scraping codes - Google does not allow it.
    if idx%100 == 0:
        print(idx)
    o_lat,o_long,d_lat,d_long = OD_df.loc[idx, ['o_lat','o_long','d_lat','d_long']]
    origin = (o_lat,o_long)
    destination = (d_lat,d_long)
    result = gmaps.distance_matrix(origin, destination, mode = 'driving')
    OD_time_dic[idx] = result

# Augment Google data
OD_from_google_api = {}
OD_from_google_api['idx'] = [] # Important for combining two dfs
OD_from_google_api['d_address'] = []
OD_from_google_api['o_address'] = []
OD_from_google_api['od_duration_text'] = []
OD_from_google_api['od_duration_value'] = []
OD_from_google_api['od_distance_text'] = []
OD_from_google_api['od_distance_value'] = []

for key in OD_time_dic.keys():
    if key%100 == 0:
        print(key)
    OD_from_google_api['idx'].append(key)
    OD_from_google_api['d_address'].append(OD_time_dic[key]['destination_addresses'][0])
    OD_from_google_api['o_address'].append(OD_time_dic[key]['origin_addresses'][0])
    OD_from_google_api['od_duration_text'].append(OD_time_dic[key]['rows'][0]['elements'][0]['duration']['text'])
    OD_from_google_api['od_duration_value'].append(OD_time_dic[key]['rows'][0]['elements'][0]['duration']['value'])
    OD_from_google_api['od_distance_text'].append(OD_time_dic[key]['rows'][0]['elements'][0]['distance']['text'])
    OD_from_google_api['od_distance_value'].append(OD_time_dic[key]['rows'][0]['elements'][0]['distance']['value'])

OD_from_google_api_df = pd.DataFrame(OD_from_google_api)

# merge
OD_merged_google_api =  OD_df.merge(OD_from_google_api_df, left_index=True, right_index=True)
OD_merged_google_api

# save
import pickle

with open("../data/OD_Google_API_raw.pickle", 'wb') as w:
    pickle.dump(OD_time_dic, w, protocol=pickle.HIGHEST_PROTOCOL)

with open("../data/OD_Google_API_With_Map_Info.pickle", 'wb') as w:
    pickle.dump(OD_merged_google_api, w, protocol=pickle.HIGHEST_PROTOCOL)










