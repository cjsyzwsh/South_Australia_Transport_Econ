# processing the SA2 and road shapefiles
# inputs: raw SA2 and road shapefiles
# Outputs: Adelaide SA2 nodal and link dataframes with transport information
#       Outputs are pickles:
#           sa2_node_with_only_transport_attributes.pickle
#           sa2_edge_with_only_transport_attributes.pickle
#       Processing files saved:
#           sa2_adelaide.shp, sa2_adelaide_edge.shp, OD_full_path.pickle, sa2_roads_in_adelaide.shp, etc.
# util needed: shortest path dictionary; a function turning the road networks to the link dataframe.
# time: ~15 min

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pysal.lib import weights
import networkx as nx
import momepy
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

# # read files
# mount_path = "/Users/shenhaowang/Dropbox (MIT)/project_econ_opportunity_south_Australia"



#region 1. Extact the SA2s for Adelaide area.
# raw data
sa2_shape = gpd.read_file(raw_data_path + "sa2/SA2_2016_AUST.shp")

# Keep Adelaide area
# info from: file:///Users/shenhaowang/Downloads/StatePublicHealthPlan_Final.pdf (page 32)
adelaide_sa4_set = ['401','402','403','404']
sa2_adelaide = sa2_shape.loc[sa2_shape.SA4_CODE16.isin(adelaide_sa4_set)]
print("Shape of SA2 in the Adelaide area is: ", sa2_adelaide.shape)

# only use the most relevant variables.
sa2_adelaide = sa2_adelaide[['SA2_MAIN16', 'SA2_NAME16', 'geometry']]

# projection
sa2_adelaide.crs = 'epsg:3112'
print(sa2_adelaide.crs)

# create a sa2_adelaide link dataframe
index = pd.MultiIndex.from_product([sa2_adelaide['SA2_MAIN16'], sa2_adelaide['SA2_MAIN16']], names=['O', 'D'])
sa2_adelaide_link_df = pd.DataFrame(index=index).reset_index()

# add the geometry part to sa2_adelaide_link_df
from shapely.geometry import LineString
edge_list = []
for idx in range(sa2_adelaide_link_df.shape[0]):
    origin = sa2_adelaide_link_df.loc[idx, 'O']
    destination = sa2_adelaide_link_df.loc[idx, 'D']
    edge = LineString([sa2_adelaide.loc[sa2_adelaide['SA2_MAIN16'] == origin, 'geometry'].centroid.values[0],
                      sa2_adelaide.loc[sa2_adelaide['SA2_MAIN16'] == destination, 'geometry'].centroid.values[0]])
    edge_list.append(edge)

sa2_adelaide_link_df['geometry'] = edge_list

# create the gpd object
sa2_adelaide_link = gpd.GeoDataFrame(sa2_adelaide_link_df, crs='epsg:3112')

# save the process SA2 Adelaide shapefile
sa2_adelaide.to_file(intermediate_data_path+'shapefiles/sa2_adelaide.shp')
sa2_adelaide_link.to_file(intermediate_data_path+'shapefiles/sa2_adelaide_edge.shp')
#endregion



#region 2. Create the OD shortest path dictionary for SA2 Adelaide shapefile.
sa2_adelaide=gpd.read_file(intermediate_data_path+'shapefiles/sa2_adelaide.shp')

# create the queen contiguity network
adelaide_queen=weights.contiguity.Queen.from_dataframe(sa2_adelaide)

# create the kernel network (using Euclidean distances)
sa2_adelaide_kernel = weights.distance.Kernel.from_dataframe(sa2_adelaide, k=109)

# turn the defaults to euclidean distances as weights.
for i in sa2_adelaide_kernel.neighbors.keys():
    for j_idx in range(len(sa2_adelaide_kernel.neighbors[i])):
        j = sa2_adelaide_kernel.neighbors[i][j_idx]
        # note that kw.weights indices are
        # i (node index), j_idx (index of the node on the list - not node index!)
        weight = sa2_adelaide_kernel.weights[i][j_idx]
        distance = (1 - weight) * sa2_adelaide_kernel.bandwidth[i]
        sa2_adelaide_kernel.weights[i][j_idx] = distance[0]

# assign euclidean weights to Queen net
for o in adelaide_queen.neighbors.keys():
#   print(o)
    for d_idx in range(len(adelaide_queen.neighbors[o])):
        d = adelaide_queen.neighbors[o][d_idx] # return the o and d SA2 original indices.
        weight = sa2_adelaide_kernel[o][d] # get the kernel weight associated with the o and d.
        adelaide_queen.weights[o][d_idx] = weight

# print(adelaide_queen.weights)

# create the nx object
adelaide_nx = adelaide_queen.to_networkx()
# assign weights to adelaide_nx
for o,d in adelaide_nx.edges:
    adelaide_nx.edges[o,d]['weight'] = adelaide_queen[o][d]

# create the OD dictionary for the full shortest paths.
path=dict(nx.all_pairs_dijkstra(adelaide_nx, weight='weight'))

# create a OD dictionary.
OD_full_path = {}

for o in range(110):
    for d in range(110):
        if d == 103 or o == 103: # note that 103 is the island - this is no path to it.
            pass
        else:
            OD_full_path[(o,d)] = path[o][1][d]

# note: OD_full_path idx is the same as sa2_adelaide!
with open(intermediate_data_path+'OD_full_path.pickle', 'wb') as f:
    pickle.dump(OD_full_path, f)
#endregion



#region 3. Read road shapefiles and save them
sa2_roads = gpd.read_file(raw_data_path + "roads/Roads_GDA2020.shp")
sa2_roads = sa2_roads.loc[~sa2_roads['class'].isna(),]

# projection to epsg:3112
sa2_roads.crs = 'epsg:3112'

# combine freeway and highway as one category (HWY).
sa2_roads.loc[sa2_roads['class'] == 'FREE', 'class'] = 'HWY'

# extract three types of roads for GIS visualization
sa2_roads_LOCL = sa2_roads.loc[sa2_roads['class'] == 'LOCL', :]
sa2_roads_HWY = sa2_roads.loc[sa2_roads['class'] == 'HWY', :]
sa2_roads_UND = sa2_roads.loc[sa2_roads['class'] == 'UND', :]

# np.unique(sa2_roads['class'], return_counts = True)


# save shapefiles
sa2_roads.to_file(intermediate_data_path+"shapefiles/sa2_roads.shp")
sa2_roads_LOCL.to_file(intermediate_data_path+"shapefiles/sa2_roads_LOCL.shp")
sa2_roads_HWY.to_file(intermediate_data_path+"shapefiles/sa2_roads_HWY.shp")
sa2_roads_UND.to_file(intermediate_data_path+"shapefiles/sa2_roads_UND.shp")

#endregion




#region 4. Turn road shapefiles to node attributes of SA2s' nodes.
# attributes: number of road counts and intersection counts.
# inputs: roads and sa2 shapefiles
# outputs: sa2 shapefile with road attributes.
sa2_roads = gpd.read_file(intermediate_data_path+"shapefiles/sa2_roads.shp")
sa2_adelaide = gpd.read_file(intermediate_data_path+'shapefiles/sa2_adelaide.shp')

# augment road class info to sa2_adelaide
sa2_adelaide_road_attributes, roads_in_adelaide = util.compute_road_attributes(sa2_adelaide, sa2_roads)
sa2_adelaide_road_attributes['num_roads'] = np.sum(sa2_adelaide_road_attributes[['class_ART', 'class_BUS', 'class_COLL',
                                                                         'class_HWY', 'class_LOCL','class_SUBA', 'class_TRK2',
                                                                         'class_TRK4', 'class_UND']], axis = 1)

# augment intersection attributes to sa2_adelaide
sa2_adelaide_intersection_attributes = util.compute_intersection_attributes(sa2_adelaide_road_attributes, roads_in_adelaide)

# merge sa2_adelaide, sa2_adelaide_road_attributes, and sa2_adelaide_intersection_attributes
sa2_adelaide_with_transport_attributes = sa2_adelaide.merge(sa2_adelaide_road_attributes, on='SA2_MAIN16', how='outer', suffixes=("","_x"))
sa2_adelaide_with_transport_attributes.drop(columns=['SA2_NAME16_x', 'geometry_x'], inplace=True)
sa2_adelaide_with_transport_attributes = sa2_adelaide_with_transport_attributes.merge(sa2_adelaide_intersection_attributes, on='SA2_MAIN16', how='outer', suffixes=("","_x"))

# save sa2_adelaide_with_transport_attributes and roads_in_adelaide
sa2_adelaide_with_transport_attributes.to_pickle(intermediate_data_path+"sa2_node_with_only_transport_attributes.pickle")
roads_in_adelaide.to_file(intermediate_data_path+"shapefiles/sa2_roads_in_adelaide.shp")
# sw: Wow. Pickle can save & read the shapefiles with crs info kept.
# sw: I still saved to shp files because QGIS cannot read pickle, I guess.
# with open("./data/sa2_adelaide_with_transport_attributes.pickle", 'rb') as f:
#     x_file = pickle.load(f)
# print(x_file.crs)

#endregion


#region 5. Turn road shapefiles to the attributes of SA2s' edges.
# It takes about five minutes for processing.
# roads_in_adelaide = gpd.read_file("./data/shapefiles/sa2_roads_in_adelaide.shp")

# 1. edge file
sa2_adelaide_edge = gpd.read_file(intermediate_data_path+'shapefiles/sa2_adelaide_edge.shp')

# 2. transport attribute file
with open(intermediate_data_path+"sa2_node_with_only_transport_attributes.pickle", 'rb') as f:
    sa2_adelaide_with_transport_attributes = pickle.load(f)

# 3. OD path file
with open(intermediate_data_path+'OD_full_path.pickle', 'rb') as f:
    OD_full_path = pickle.load(f)

# add the road and intersection attributes to the sa2_adelaide_edge data set.
attribute_name_list = ['class_ART', 'class_BUS', 'class_COLL',
                       'class_HWY', 'class_LOCL', 'class_SUBA',
                       'class_TRK2', 'class_TRK4', 'class_UND', 'num_roads', 'num_nodes', 'num_1degree',
                       'num_2degree', 'num_3degree', 'num_4degree', 'num_greater5degree']

sa2_adelaide_edge[attribute_name_list] = 0.0 # init values

# add road and intersection attributes to the edge df.
for idx in np.arange(sa2_adelaide_edge.shape[0]):
    if idx%1000 == 0:
        print(idx)
    origin = sa2_adelaide_edge.loc[idx, 'O']
    destination = sa2_adelaide_edge.loc[idx, 'D']
    o_idx = sa2_adelaide_with_transport_attributes.index[sa2_adelaide_with_transport_attributes.SA2_MAIN16 == origin].tolist()[0]
    d_idx = sa2_adelaide_with_transport_attributes.index[sa2_adelaide_with_transport_attributes.SA2_MAIN16 == destination].tolist()[0]
    # print(o_idx,d_idx)

    try:
        # OD_full_path might not have all the shortest path...
        # note that the OD_full_path idx is consistent with sa2_adelaide.
        idx_list_on_shortest_path = OD_full_path[(o_idx, d_idx)]
        for node_on_shortest_path in idx_list_on_shortest_path:
            sa2_adelaide_edge.loc[idx, attribute_name_list] += sa2_adelaide_with_transport_attributes.loc[
                node_on_shortest_path, attribute_name_list]
    except KeyError as error:
        pass

# output two pickles:
# node network with transport info: sa2_adelaide_with_transport_attributes
# edge network with transport info: sa2_adelaide_edge
sa2_adelaide_with_transport_attributes.to_pickle(intermediate_data_path+'sa2_node_with_only_transport_attributes.pickle')
sa2_adelaide_edge.to_pickle(intermediate_data_path+'sa2_edge_with_only_transport_attributes.pickle')

#endregion



















