import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot
from pysal.lib import weights
import networkx as nx
from scipy.spatial import distance
import momepy
import pickle
import math
import sys

def shortest_path(shp_file):
    """
        Inputs:
            shp_file - a shp file
        Outputs:
            A dictionary that maps (o,d) pairs to its shortest path
    """
    
    print("=====Running shortest_path=====")
    
    #convert to austrailia projection
    shp_file_proj = shp_file.to_crs("epsg:3112")
    # Step 1. Queen net
    shp_file_queen = weights.contiguity.Queen.from_dataframe(shp_file)
    
    # Step 2. Kernel net with the right euclidean weighting
    #Use all K nearest neighbors
    shp_file_kernel = weights.distance.Kernel.from_dataframe(shp_file_proj, k = shp_file_queen.n - 1)
    # turn the defaults to euclidean distances as weights.
    for i in shp_file_kernel.neighbors.keys():
        for j_idx in range(len(shp_file_kernel.neighbors[i])):
            j = shp_file_kernel.neighbors[i][j_idx]
            # note that kw.weights indices are 
            # i (node index), j_idx (index of the node on the list - not node index!)
            weight = shp_file_kernel.weights[i][j_idx]
            distance = (1 - weight)*shp_file_kernel.bandwidth[i]
            shp_file_kernel.weights[i][j_idx] = distance[0]
    
    # Step 3. assign euclidean weights to Queen net
    for o in shp_file_queen.neighbors.keys():
        for d_idx in range(len(shp_file_queen.neighbors[o])):
            d = shp_file_queen.neighbors[o][d_idx] # return the o and d SA2 original indices. 
            weight = shp_file_kernel[o][d] # get the kernel weight associated with the o and d.
            shp_file_queen.weights[o][d_idx] = weight
            
            
    # create the queen network in nx
    shp_file_nx = shp_file_queen.to_networkx()

    # assign weights to adelaide_nx
    for o,d in shp_file_nx.edges:
        shp_file_nx.edges[o,d]['weight'] = shp_file_queen[o][d]

    # example weight between nodes 0 and 1.
    shp_file_nx.get_edge_data(0, 1)
    
    # full paths.
    # return: (node, (distance, path))
    path=dict(nx.all_pairs_dijkstra(shp_file_nx, weight='weight'))
    
    # create a OD dictionary.
    OD_full_path = {}

    for o in range(110):
        for d in range(110):
            if d==103 or o==103: # note that 103 is the island - this is no path to it.
                pass
            else:
                OD_full_path[(o,d)] = path[o][1][d]
                
    print("=====DONE shortest_path=====")        
        
    return OD_full_path

def union_road_land_shp(shp, road_shp):
    """
        Inputs:
            shp - a shp file
            road_shp - the shp file containing road information
        Outputs:
            A shp file with all the information merged
    """
    
    print("=====Running union_road_land_shp=====")
    
    # crs and projection
    shp_proj = shp.to_crs("epsg:3112")
    sa2_roads_proj = road_shp.to_crs("epsg:3112")

    
    # create the centroids for roads
    road_centroid = sa2_roads_proj.centroid
    
    # attach SA2 idx to road networks
    sa2_roads_proj['SA2_loc'] = -1 # init as -1.

    for SA2_idx in range(shp_proj.shape[0]):
        # assign SA2_idx to the road network
        within_logic = road_centroid.within(shp_proj.loc[SA2_idx, 'geometry'])
        sa2_roads_proj.loc[within_logic, 'SA2_loc'] = SA2_idx
        
    # Use only the 'class' variable for now. 
    sa2_roads_class_proj = sa2_roads_proj[['class', 'geometry', 'SA2_loc']]
    sa2_roads_class_proj_dummies = pd.get_dummies(sa2_roads_class_proj)
    
    
    # aggregate the road attribute dummies for SA2.
    sa2_roads_class_proj_dummies = sa2_roads_class_proj_dummies.loc[sa2_roads_class_proj_dummies['SA2_loc'] > -1]
    sa2_road_class_agg=sa2_roads_class_proj_dummies.groupby(by='SA2_loc').sum()
    
    # augment road class variables to SA2_network.
    shp_proj = shp_proj.merge(sa2_road_class_agg, how='inner', left_index=True, right_index=True)
    
    print("=====DONE union_road_land_shp=====")
    
    return shp_proj, sa2_roads_proj

def get_degree_df(shp_proj, road_proj):
    """
        Inputs:
            shp_proj - a shp projection merged with road info; the first output union_road_land_shp(shp, road_shp)
            road_proj - a shp projection that has only road information; the second output of union_road_land_shp(shp, road_shp)
        Outputs:
            degree_df - a df with SA2 code and its degree counts
            node_degree_df - a pickle file with the shp file information + node degree counts
            edge_degree_df - a pickle file with the shp file information + agg node degree counts of the shortest path
    """
    
    print("=====Running get_degree_df=====")
    
    
    #get counts
    count ={}
    for elt in road_proj["SA2_loc"]:
        if elt in count:
            count[elt] += 1
        else:
            count[elt] = 1
    SA_idxs = sorted((key,count[key]) for key in count)
    
    
    sa_idx_to_graph = {}
    for sa_idx,c in SA_idxs[1:]:
        within = road_proj[road_proj["SA2_loc"]==sa_idx]
        graph = momepy.gdf_to_nx(within, approach='primal')
        sa_idx_to_graph[sa_idx] = graph
        
    
    degree_df = pd.DataFrame(columns=["SA2_MAIN16", "num_nodes", 
                                      "num_1degree", "num_2degree", 
                                      "num_3degree", "num_4degree", 
                                      "num_greater5degree"])
    
    for sa_idx in sa_idx_to_graph:
        g = sa_idx_to_graph[sa_idx]
        degree = dict(nx.degree(g))
        nx.set_node_attributes(g, degree, 'degree')
        g = momepy.node_degree(g, name='degree')
        node_df, edge_df, sw = momepy.nx_to_gdf(g, points=True, lines=True,
                                        spatial_weights=True)

        SA2_MAIN16 = shp_proj.iloc[sa_idx]["SA2_MAIN16"]
        #nodes is intersections
        num_nodes = len(node_df)
        #num_0degree = len(node_df[node_df["degree"]==0])
        num_1degree = len(node_df[node_df["degree"]==1])
        num_2degree = len(node_df[node_df["degree"]==2])
        num_3degree = len(node_df[node_df["degree"]==3])
        num_4degree = len(node_df[node_df["degree"]==4])
        num_greater5degree = len(node_df[node_df["degree"]>=5])
        degree_df = degree_df.append({"SA2_MAIN16": SA2_MAIN16, "num_nodes":num_nodes,  
                                      "num_1degree":num_1degree, "num_2degree":num_2degree, "num_3degree":num_3degree,
                                      "num_4degree":num_4degree,
                                      "num_greater5degree":num_greater5degree},
                                    ignore_index=True)
        
    print("=====DONE degree df=====")
    
    return degree_df

def get_specific_df(OD_full_path, shp, shp_proj, mount_path, sa4_set=['401','402','403','404']):
    """
        Inputs:
            OD_full_path - output from shortest_path(); the first output union_road_land_shp(shp, road_shp)
            shp - original shp file
            shp_proj - the shp file merged with road attributes 
            
        Outputs:
            edge_specific_df - initial edge df with all info
            node_specific_df - intial node df with all info
    """
    
    print("=====Running get_specific_df=====")
    
    # read google api info
    with open(mount_path + '/SA data/dataSA/OD_Google_API_raw.pickle', 'rb') as w:
        OD_google_raw = pickle.load(w)

    with open(mount_path + '/SA data/dataSA/OD_Google_API_With_Map_Info.pickle', 'rb') as w:
        OD_google_with_map = pickle.load(w)
        
    jobs_all_sub = jobs_all[['num_jobs_000_persons', 'sa2_code16', 'median_income_per_job_aud_persons']]
    
    flow_adelaide_df = flow_df.loc[np.array([x[:3] in sa4_set for x in flow_df.agent_home_sa2])]
    flow_adelaide_df = flow_adelaide_df.loc[np.array([x[:3] in sa4_set for x in flow_adelaide_df.sa2])]
    
    flow_adelaide_df.rename(columns={'agent_home_sa2':'origin','sa2':'destination'}, inplace=True)
    flow_adelaide_df['OD'] = ''
    flow_adelaide_df['OD'] = flow_adelaide_df['origin'] + flow_adelaide_df['destination']
    flow_adelaide_df.groupby(by='OD').sum() # no repetition. 
    
    # reindex
    flow_adelaide_df.index = np.arange(flow_adelaide_df.shape[0])
    
    # create ten columns here.
    road_attribute_names_list = ['class_ART', 'class_BUS', 'class_COLL',
                                 'class_FREE', 'class_HWY', 'class_LOCL', 'class_SUBA', 'class_TRK2',
                                 'class_TRK4', 'class_UND']
    flow_adelaide_df[road_attribute_names_list] = 0.0
    
    
    # add the road attributes on the shortest path to the flow_adelaide_df.
    # time cost: 3-5 mins?
    for idx in np.arange(flow_adelaide_df.shape[0]):
        origin = flow_adelaide_df.loc[idx, 'origin']
        destination = flow_adelaide_df.loc[idx, 'destination']
        o_idx = shp.index[shp.SA2_MAIN16==origin].tolist()[0]
        d_idx = shp.index[shp.SA2_MAIN16==destination].tolist()[0]
        #print(o_idx,d_idx)

        try:
            # OD_full_path might not have all the shortest path...
            idx_list_on_shortest_path = OD_full_path[(o_idx, d_idx)]
            for node_on_shortest_path in idx_list_on_shortest_path:
                flow_adelaide_df.loc[idx, road_attribute_names_list] += shp_proj.loc[node_on_shortest_path, road_attribute_names_list]        
        except KeyError as error:
            pass
        
    
    # add the job information to flow dataframe.
    # origin
    flow_adelaide_df=flow_adelaide_df.merge(jobs_all_sub, left_on='origin', right_on='sa2_code16', how = 'left')
    flow_adelaide_df=flow_adelaide_df.rename(columns={'num_jobs_000_persons':'num_jobs_000_persons_origin', 'median_income_per_job_aud_persons':'median_income_per_job_aud_origin'})

    # destination
    flow_adelaide_df=flow_adelaide_df.merge(jobs_all_sub, left_on='destination', right_on='sa2_code16', how = 'left')
    flow_adelaide_df=flow_adelaide_df.rename(columns={'num_jobs_000_persons':'num_jobs_000_persons_destination', 'median_income_per_job_aud_persons':'median_income_per_job_aud_destination'})

    
    # augment the travel time and distance information to flow_adelaide_df
    flow_adelaide_df['od_duration_value']=0.0 
    flow_adelaide_df['od_distance_value']=0.0 

    for idx in range(flow_adelaide_df.shape[0]):
        if idx%100 == 0:
            print(idx)

        # idx is the index in flow_adelaide_df
        origin_sa2_idx = flow_adelaide_df.loc[idx,'origin']
        destination_sa2_idx = flow_adelaide_df.loc[idx,'destination']

        # return the corresponding idx from OD_Google_API
        filter_idx = np.multiply(OD_google_with_map.loc[:, 'o_sa2_idx'] == origin_sa2_idx,
                                 OD_google_with_map.loc[:, 'd_sa2_idx'] == destination_sa2_idx)
        idx_google_api = OD_google_with_map.index[filter_idx].tolist()[0] # this is the index in OD_google_with_map

        # 
        flow_adelaide_df.loc[idx, 'od_duration_value'] = OD_google_with_map.loc[idx_google_api, 'od_duration_value']
        flow_adelaide_df.loc[idx, 'od_distance_value'] = OD_google_with_map.loc[idx_google_api, 'od_distance_value']
        
    # replace 0.0 values by 1.0
    cols = ['sum_stay_duration','unique_agents','total_stays',
            'class_ART', 'class_BUS', 'class_COLL', 'class_FREE', 'class_HWY', 'class_LOCL', 
            'class_SUBA', 'class_TRK2', 'class_TRK4', 'class_UND',
            'od_duration_value', 'od_distance_value']

    for col in cols:
        flow_adelaide_df.loc[flow_adelaide_df.loc[:,col] == 0.0, col] = 1.0
        
    # dropped 433 observations. The df has nan.
    flow_adelaide_df.dropna(how = 'any', inplace = True)
    
    # add total road count as a variable
    flow_adelaide_df['road_counts'] = np.sum(flow_adelaide_df[['class_ART', 'class_BUS', 'class_COLL', 'class_FREE', 'class_HWY', 'class_LOCL', 
                                         'class_SUBA', 'class_TRK2', 'class_TRK4', 'class_UND']], axis = 1)
    edge_specific_df = flow_adelaide_df.copy()
    
    print("=====DONE EDGE=====")
    
    # origin and destination flow counts
    origin_flow_counts = flow_adelaide_df.groupby(by="origin",as_index=False,sort=False).sum()[['origin','unique_agents','sum_stay_duration','total_stays']]
    destination_flow_counts = flow_adelaide_df.groupby(by="destination",as_index=False,sort=False).sum()[['destination','unique_agents','sum_stay_duration','total_stays']]
    
    # compute origin and destination entropy (w.r.t. location). flow location diversity.
    # origin
    origin_flow_count_n = flow_adelaide_df.groupby('origin')[['unique_agents','sum_stay_duration','total_stays']].transform('sum')
    values = flow_adelaide_df[['unique_agents','sum_stay_duration','total_stays']]/origin_flow_count_n
    flow_adelaide_df[['unique_agents_origin_entropy','sum_stay_duration_origin_entropy','total_stays_origin_entropy']] = \
        -(values*np.log(values))
    origin_flow_entropy=flow_adelaide_df.groupby('origin',as_index=False,sort=False)[['unique_agents_origin_entropy','sum_stay_duration_origin_entropy','total_stays_origin_entropy']].sum()

    # destination
    destination_flow_count_n = flow_adelaide_df.groupby('destination')[['unique_agents','sum_stay_duration','total_stays']].transform('sum')
    values = flow_adelaide_df[['unique_agents','sum_stay_duration','total_stays']]/destination_flow_count_n
    flow_adelaide_df[['unique_agents_destination_entropy','sum_stay_duration_destination_entropy','total_stays_destination_entropy']] = \
        -(values*np.log(values))
    destination_flow_entropy=flow_adelaide_df.groupby('destination',as_index=False,sort=False)[['unique_agents_destination_entropy','sum_stay_duration_destination_entropy','total_stays_destination_entropy']].sum()

    # merge data to sa2_adelaide_road_proj
    # augment income and jobs
    sa2_data_prep=pd.merge(shp_proj, jobs_all_sub, left_on='SA2_MAIN16', right_on='sa2_code16', how = 'inner')
    sa2_data_prep=pd.merge(sa2_data_prep, origin_flow_counts, left_on='SA2_MAIN16', right_on='origin', how='inner', suffixes=[None,'_origin_counts'])
    sa2_data_prep=pd.merge(sa2_data_prep, destination_flow_counts, left_on='SA2_MAIN16', right_on='destination', how='inner', suffixes=[None,'_destination_counts'])
    sa2_data_prep=pd.merge(sa2_data_prep, origin_flow_entropy, left_on='SA2_MAIN16', right_on='origin', how='inner')
    sa2_data_prep=pd.merge(sa2_data_prep, destination_flow_entropy, left_on='SA2_MAIN16', right_on='destination', how='inner')

    # rename the '_origin_counts'
    sa2_data_prep = sa2_data_prep.rename(columns={'unique_agents':'unique_agents_origin_counts',
                                          'sum_stay_duration':'sum_stay_duration_origin_counts',
                                          'total_stays':'total_stays_origin_counts'})
    
    node_specific_df = sa2_data_prep.copy()
    
    print("=====DONE get_specific_df=====")
    
    return edge_specific_df, node_specific_df

def union_degree(node_df, edge_df, degree_df, OD_full_path):
    """
        Inputs:
            node_df, edge_df - the node and edge specific df from get_specific_df
            OD_full_path - output of shortest_path
            degree_df - output of get degree df
        Outputs:
            edge_degree_df, node_degree_df - respective dfs merged with degree df
    """
    
    print("=====Running union_degree=====")
    
    node_degree_df = node_df.merge(degree_df, how="left", on="SA2_MAIN16")
    
    origin_dest = list(zip(edge_df["origin"].values, edge_df["destination"].values))
    
    edge_degree_df = pd.DataFrame(columns=["sa2_code16_x", "sa2_code16_y", "num_nodes_x", 
                                  "num_1degree_x", "num_2degree_x", "num_3degree_x", "num_4degree_x",
                                    "num_greater5degree_x",
                                      "num_nodes_y", 
                                  "num_1degree_y", "num_2degree_y", "num_3degree_y", "num_4degree_y",
                                    "num_greater5degree_y"])
    
    sa_to_i = {}
    i_to_sa = {}
    sa_to_data = {}
    for i, row in degree_df.iterrows():
        print(i)
        i_to_sa[i] = row["SA2_MAIN16"]
        sa_to_i[row["SA2_MAIN16"]] = i
        sa_to_data[row["SA2_MAIN16"]] = row[['num_nodes', 'num_1degree','num_2degree', 'num_3degree', 'num_4degree', 'num_greater5degree']]
        
    for o,d in origin_dest:
        if o != d:
            o_data = degree_df[degree_df["SA2_MAIN16"]==o]
            d_data = degree_df[degree_df["SA2_MAIN16"]==d]

            num_nodes_pth = 0
            num_1degree_pth = 0
            num_2degree_pth = 0
            num_3degree_pth = 0
            num_4degree_pth = 0
            num_greater5degree_pth = 0
            oid = sa_to_i[o]
            did = sa_to_i[d]
            for i in OD_full_path[(oid,did)]:
                sa = i_to_sa[i]
                num_nodes_pth += float(sa_to_data[sa][0])
                num_1degree_pth += float(sa_to_data[sa][1])
                num_2degree_pth += float(sa_to_data[sa][2])
                num_3degree_pth += float(sa_to_data[sa][3])
                num_4degree_pth += float(sa_to_data[sa][4])
                num_greater5degree_pth += float(sa_to_data[sa][5])



            num_nodes_x = float(o_data["num_nodes"].iloc[0])
            num_1degree_x = float(o_data["num_1degree"].iloc[0])
            num_2degree_x = float(o_data["num_2degree"].iloc[0])
            num_3degree_x = float(o_data["num_3degree"].iloc[0])
            num_4degree_x = float(o_data["num_4degree"].iloc[0])
            num_greater5degree_x = float(o_data["num_greater5degree"].iloc[0])

            num_nodes_y = float(d_data["num_nodes"].iloc[0])
            num_1degree_y = float(d_data["num_1degree"].iloc[0])
            num_2degree_y = float(d_data["num_2degree"].iloc[0])
            num_3degree_y = float(d_data["num_3degree"].iloc[0])
            num_4degree_y = float(d_data["num_4degree"].iloc[0])
            num_greater5degree_y = float(d_data["num_greater5degree"].iloc[0])


        else:
            o_data = degree_df[degree_df["SA2_MAIN16"]==o]
            d_data = degree_df[degree_df["SA2_MAIN16"]==d]
            num_nodes_x = num_nodes_y = num_nodes_pth = float(o_data["num_nodes"].iloc[0])
            num_1degree_x = num_1degree_y = num_1degree_pth = float(o_data["num_1degree"].iloc[0])
            num_2degree_x = num_2degree_y = num_2degree_pth = float(o_data["num_2degree"].iloc[0])
            num_3degree_x = num_3degree_y = num_3degree_pth = float(o_data["num_3degree"].iloc[0])
            num_4degree_x = num_4degree_y = num_4degree_pth = float(o_data["num_4degree"].iloc[0])
            num_greater5degree_x = num_greater5degree_y = num_greater5degree_pth = float(o_data["num_greater5degree"].iloc[0])

        edge_degree_df = edge_degree_df.append({"sa2_code16_x": o, "sa2_code16_y":d ,"num_nodes_x":num_nodes_x, 
                                      "num_1degree_x":num_1degree_x, "num_2degree_x":num_2degree_x, 
                                      "num_3degree_x":num_3degree_x, "num_4degree_x":num_4degree_x,
                                      "num_greater5degree_x":num_greater5degree_x,
                                      "num_nodes_y":num_nodes_y, 
                                      "num_1degree_y":num_1degree_y, "num_2degree_y":num_2degree_y, 
                                      "num_3degree_y":num_3degree_y, "num_4degree_y":num_4degree_y,
                                      "num_greater5degree_y":num_greater5degree_y,
                                      "num_nodes_pth":num_nodes_pth,
                                      "num_1degree_pth":num_1degree_pth,
                                      "num_2degree_pth":num_2degree_pth,
                                      "num_3degree_pth":num_3degree_pth,
                                      "num_4degree_pth":num_4degree_pth,
                                       "num_greater5degree_pth":num_greater5degree_pth        },
                                    ignore_index=True)
    edge_degree_df = edge_df.merge(edge_degree_df, how="left", on=["sa2_code16_x","sa2_code16_y"])
    
    print("=====DONE union_degree=====")
    
    return edge_degree_df, node_degree_df

def union_poi(node_degree_df, edge_degree_df):
    """
        Inputs:
            node_degree_df, edge_degree_df - outputs of union degree
        Outputs:
            Inputs merged with poi df
    """
    
    print("=====Running union_poi=====")
    
    
#     poi_df = pd.read_pickle("../../data_process/poi_df.pickle")
#     sa_codes_poi = []
#     for i, centroid in enumerate(poi_df.geometry):
#         if i%100 == 0: print(i)
#         found = False
#         for i, row in sa2_south_au.iterrows():
#             if row["geometry"].contains(centroid):
#                 sa_codes_poi.append(row["SA2_MAIN16"])
#                 found = True
#                 break
#         if not found:
#             sa_codes_poi.append("0")
#     poi_df["SA2_MAIN16"] = sa_codes_poi
#     poi_df.to_pickle("../../data_process/poi_df_cleaned.pickle")
    poi_df = pd.read_pickle("../../data_process/poi_df_cleaned.pickle")
    poi_df = poi_df[poi_df["SA2_MAIN16"]!="0"]
    
    
    count = poi_df.groupby(["SA2_MAIN16"],as_index=False).aggregate(["count"])
    split_count = poi_df.groupby(["SA2_MAIN16","type"],as_index=False).aggregate(["count"])
    
    poi_df = pd.DataFrame()
    
    poi_df["SA2_MAIN16"] = count.index.values
    poi_df["poi_count"] = count[( 'geometry', 'count')].values
    
    entropy = {}
    for i, row in split_count.iterrows():
        sa_id, _type = i
        total_count = poi_df.loc[poi_df["SA2_MAIN16"]==sa_id]["poi_count"]
        val = row[( 'geometry', 'count')]/total_count

        if sa_id not in entropy:
            entropy[sa_id] = (-val * np.log(val))
        else:
            entropy[sa_id] += (-val * np.log(val))
            
    entropy_list = []
    for sa_id in poi_df.SA2_MAIN16:
        entropy_list.append(float(entropy[sa_id]))
        
    poi_df["poi_count_entropy"] = entropy_list
    
    node_degree_entropy_df = node_degree_df.merge(poi_df,how="left",on="SA2_MAIN16")
    
    sa_ids_poi = set(poi_df["SA2_MAIN16"].values)
    
    edge_degree_df = edge_degree_df[edge_degree_df["sa2_code16_y"].isin(sa_ids_poi)]
    edge_degree_df = edge_degree_df[edge_degree_df["sa2_code16_x"].isin(sa_ids_poi)]
    
    count_dic = {key:val for key,val in zip(poi_df["SA2_MAIN16"].values, poi_df["poi_count"].values)}
    
    entropy_x = []
    poi_count_x = []
    notin = 0
    for sa_id in edge_degree_df["sa2_code16_x"].values:
        if sa_id in entropy:
            entropy_x.append(float(entropy[sa_id]))
            poi_count_x.append(float(count_dic[sa_id]))
        else:
            notin += 1
    notin=0
    entropy_y = []
    poi_count_y = []
    for sa_id in edge_degree_df["sa2_code16_y"]:
        if sa_id in entropy:
            entropy_y.append(float(entropy[sa_id]))
            poi_count_y.append(float(count_dic[sa_id]))
        else:
            notin += 1
    
    edge_degree_df["poi_entropy_x"] = entropy_x
    edge_degree_df["poi_entropy_y"] = entropy_y
    
    edge_degree_df["poi_count_x"] = poi_count_x
    edge_degree_df["poi_count_y"] = poi_count_y
    
    print("=====DONE union_poi=====")
    
    return edge_degree_df, node_degree_entropy_df

def union_social(node_degree_poi_df, mount_path):
    """
        Input:
            node_degree_poi_df - output from union_poi
        Output:
            A merged df with input and social economic info
    """
    
    print("=====Running union_social=====")
    
    age_df = pd.read_csv(mount_path + "SA data/data_age.csv")
    gender_educ_df = pd.read_csv(mount_path + "SA data/data_gender_educ.csv")
    
    
    age_df = age_df[[' sa2_main16','p_tot_75_84_yrs', ' p_tot_35_44_yrs', ' p_tot_45_54_yrs',' p_tot_25_34_yrs', ' p_tot_85ov',
       ' p_tot_65_74_yrs', ' p_tot_20_24_yrs', ' p_tot_15_19_yrs',' p_tot_55_64_yrs', ' p_tot_tot']]
    
    for col in ['p_tot_75_84_yrs', ' p_tot_35_44_yrs', ' p_tot_45_54_yrs',' p_tot_25_34_yrs', ' p_tot_85ov',
       ' p_tot_65_74_yrs', ' p_tot_20_24_yrs', ' p_tot_15_19_yrs',' p_tot_55_64_yrs']:
        age_df[col] = age_df[col] / age_df[" p_tot_tot"]
        
    gender_educ_df[" gender_tot_tot"] = gender_educ_df[" m_tot_tot"] + gender_educ_df[" f_tot_tot"]
    
    for col in ['m_tot_75_84_yr', ' m_adv_dip_dip_total', ' m_tot_35_44_yr',
           ' m_tot_55_64_yr', ' f_b_deg_tot', ' m_tot_85_yr_over',
           ' f_tot_55_64_yr', ' f_cer_tot_tot', ' f_tot_65_74_yr',
           ' f_tot_15_24_yr', ' m_grad_dip_cer_tot', 
           ' f_tot_75_84_yr', ' f_tot_45_54_yr', ' m_tot_45_54_yr',
           ' f_adv_dip_dip_total', ' f_tot_35_44_yr', ' m_tot_25_34_yr',
           ' m_pg_deg_tot', ' m_tot_65_74_yr', ' m_tot_15_24_yr',
           ' f_pguate_deg_tot', ' m_b_deg_tot', ' m_cer_tot_tot',
           ' f_tot_25_34_yr', ' f_grad_dip_cer_tot', 
           ' f_tot_85_yr_over']:
        g = col.split("_")[0]
        if g in {" m", "m"}:
            gender_educ_df[col] = gender_educ_df[col] / gender_educ_df[" m_tot_tot"]
        else:
            gender_educ_df[col] = gender_educ_df[col] / gender_educ_df[" f_tot_tot"]
            
    gender_educ_df[" m_percent"] = gender_educ_df[" m_tot_tot"]/gender_educ_df[" gender_tot_tot"]
    gender_educ_df[" f_percent"] = gender_educ_df[" f_tot_tot"]/gender_educ_df[" gender_tot_tot"]
    age_gender_educ_df = age_df.merge(gender_educ_df, on=" sa2_main16", how="left")
    helper = [str(elt) for elt in age_gender_educ_df[" sa2_main16"].values]
    
    age_gender_educ_df["SA2_MAIN16"] = helper
    
    avg_med_age = []
    for idx, row in age_gender_educ_df.iterrows():
        avg = 0
        for col in ['p_tot_75_84_yrs', ' p_tot_35_44_yrs',
                   ' p_tot_45_54_yrs', ' p_tot_25_34_yrs', ' p_tot_85ov',
                   ' p_tot_65_74_yrs', ' p_tot_20_24_yrs', ' p_tot_15_19_yrs',
                   ' p_tot_55_64_yrs']:
            if col != " p_tot_85ov":
                    temp = col.split("_")
                    x = int(temp[2])
                    y = int(temp[3])
                    med = (x+y)/2
                    avg += (row[col]*med)
            else:
                    avg += (row[col]*85)
        avg_med_age.append(avg)
        
    age_gender_educ_df[" avg_med_age"] = avg_med_age
    useful_df = age_gender_educ_df[["SA2_MAIN16"," m_percent", " f_percent", " avg_med_age", " p_tot_tot"]]
    all_educ_df = pd.read_csv(mount_path + "SA data/data_all_educ.csv")
    
    degree_percentage = (all_educ_df[" p_b_deg_tot"] + all_educ_df[" p_grad_dip_cer_tot"])/all_educ_df[" p_tot_tot"]
    all_educ_df["degree_percentage"] = degree_percentage
    all_educ_df["SA2_MAIN16"] = [str(elt) for elt in all_educ_df[" sa2_main16"].values]
    
    useful_df = useful_df.merge(all_educ_df[["SA2_MAIN16","degree_percentage", " p_cer_tot_tot"]],how="left", on="SA2_MAIN16")
    useful_df.dropna()
    
    node_degree_poi_df = node_degree_poi_df.merge(useful_df, how="left", on="SA2_MAIN16")
    
    print("=====Done union_social=====")
    
    return node_degree_poi_df

def get_final_node_edge_dfs(shp_file, mount_path):

    """
        Converts shp_file (input) to the processed node_df and edge_df
    """

    ## read files

    trans_mcc_df = pd.read_pickle("../../data_process/trans_mcc_df.pkl")
    trans_age_df = pd.read_pickle("../../data_process/trans_age_df.pkl")
    flow_df = pd.read_pickle("../../data_process/flow_df.pkl")

    # read spatial files
    sa2_south_au = gpd.read_file("../../data_process/shapefiles/sa2_south_au.shp")

    # read road networks
    sa2_roads = gpd.read_file("../../data_process/shapefiles/sa2_roads.shp")

    # read job and income data
    jobs_all = pd.read_pickle("../../data_process/jobs_all.pkl")
    jobs_industries = pd.read_pickle("../../data_process/jobs_industries.pkl")

    OD_full_path = shortest_path(shp_file)

    shp_proj, sa2_roads_proj = union_road_land_shp(shp_file, sa2_roads)
    
    degree_df = get_degree_df(shp_proj,sa2_roads_proj)

    edge_specific_df, node_specific_df = get_specific_df(OD_full_path, shp_file, shp_proj, mount_path)

    edge_degree_df, node_degree_df = union_degree(node_specific_df, edge_specific_df, degree_df, OD_full_path)

    edge_degree_poi_df, node_degree_poi_df = union_poi(node_degree_df, edge_degree_df)

    node_degree_poi_social_df = union_social(node_degree_poi_df, mount_path)

    return node_degree_poi_social_df, edge_degree_poi_df

global trans_mcc_df
global trans_age_df
global flow_df
global sa2_south_au
global sa2_roads
global jobs_all
global jobs_industries
trans_mcc_df = pd.read_pickle("../../data_process/trans_mcc_df.pkl")
trans_age_df = pd.read_pickle("../../data_process/trans_age_df.pkl")
flow_df = pd.read_pickle("../../data_process/flow_df.pkl")

# read spatial files
sa2_south_au = gpd.read_file("../../data_process/shapefiles/sa2_south_au.shp")

# read road networks
sa2_roads = gpd.read_file("../../data_process/shapefiles/sa2_roads.shp")

# read job and income data
jobs_all = pd.read_pickle("../../data_process/jobs_all.pkl")
jobs_industries = pd.read_pickle("../../data_process/jobs_industries.pkl")
if __name__ == "__main__":
    #TODO:
    if len(sys.argv) != 1:
        print("Invalid arugments")
        pass
    else:
        shp_file = gpd.read_file(sys.argv[1])
        get_final_node_edge_dfs(shp_file)
        pass