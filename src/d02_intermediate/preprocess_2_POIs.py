# processing POI data
# inputs: raw POI
# outputs: node df with POIs' counts, entropy, etc.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import statsmodels.api as sm

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

# mount_path = "/Users/shenhaowang/Dropbox (MIT)/project_econ_opportunity_south_Australia/"

# read poi
poi_df = gpd.read_file(raw_data_path + "points/points.shp")
sa2_adelaide = gpd.read_file(intermediate_data_path + 'shapefiles/sa2_adelaide.shp')

# print(poi_df.shape)
# print(poi_df.columns)

#region 1. Assign POIs to SA2 (15 min)
# assign POIs to sa2 in Adelaide
sa_codes_poi = []
for i, centroid in enumerate(poi_df.geometry):
    if i % 1000 == 0: print(i)
    found = False
    for j, row in sa2_adelaide.iterrows():
        if row["geometry"].contains(centroid):
            sa_codes_poi.append(row["SA2_MAIN16"])
            found = True
            break
    if not found:
        sa_codes_poi.append("0")

poi_df["SA2_MAIN16"] = sa_codes_poi
poi_df = poi_df.loc[poi_df["SA2_MAIN16"]!='0', :]
print(poi_df.shape)

# new mapping.
drop_list = ['Historical_House',
             'adit','alpine_hut','animal_boarding','antenna','apartment','beacon','bench','bicycle_parking',
             'buffer_stop','building','bus_station','bus_stop',
             'chalet', 'charging_station', 'chimney', 'clock',
             'communications_t','compressed_air','construction','crossing',
             'device_charging_','disused','dojo', 'drinking_water',
             'elevator',
             'flagpole','fuel','funeral_home',
             'garbage_can','give_way','goods_conveyor','guest_house;hote','grave_yard','guest_house','guest_house;hote',
             'halt',
             'kiln',
             'lamp','level_crossing','loading_dock',
             'manhole', 'mast', 'milestone', 'mine', 'mine_shaft', 'motorway_junctio',
             'parking', 'parking_entrance', 'parking_space', 'proposed',
             'rest_area',
             'sanitary_dump_st', 'silo', 'speed_camera','station','steps','stop','street_cabinet',
             'street_lamp','subway_entrance','surveillance','survey_point','swings','switch',
             'tank','taxi','tomb','tower',
             'traffic_signals','traffic_signals;','trailhead','tram_level_cross','tram_stop','tree','turning_circle','turning_loop','turntable',
             'waste_basket','waste_transfer_s','wastewater_plant','water_point','water_tank','water_tap',
             'water_tower','water_well','windmill','windpump','wreck',
             'yes']
education = ['childcare', 'college', 'community_centre', 'kindergarten', 'music_school', 'school', 'university']
tour = ['attraction', 'castle', 'ferry_terminal', 'monument',
        'picnic_site', 'place_of_worship', 'viewpoint', 'zoo']
restaurant = ['bar', 'bbq', 'cafe', 'fast_food', 'food_court', 'ice_cream', 'pub', 'restaurant', 'restaurant;bar']
culture = ['arts_centre', 'artwork', 'gallery', 'library', 'memorial', 'museum',
           'piano', 'public_bookcase', 'ruins', 'shower', 'studio', 'theatre']
recreation = ['camp_pitch', 'camp_site', 'caravan_site', 'cinema', 'events_venue',
              'fountain', 'nightclub', 'stripclub', 'swimming_pool']
small_business = ['bicycle_rental', 'bicycle_repair_s', 'brothel', 'car_rental', 'car_wash', 'gambling', 'makerspace','marketplace',
                  'vending_machine','veterinary','winery']
hotel = ['hostel', 'hotel', 'motel']
information = ['information', 'monitoring_stati', 'newsagency', 'telephone']
government = ['bureau_de_change', 'courthouse', 'fire_station', 'police', 'post_box', 'post_office', 'prison', 'pumping_station',
              'recycling', 'scout_hall', 'scrapyard', 'shelter', 'shelter;drinking', 'social_facility', 'storage_tank',
              'toilets','townhall']
medical = ['clinic', 'dentist', 'doctors', 'first_aid', 'hospital', 'pharmacy', 'surgery']
finance = ['atm', 'bank']

# replacement dictionary
replacement_dict = dict(zip(drop_list+education+tour+restaurant+culture+recreation+small_business+hotel+information+government+medical+finance,
                            ['drop']*len(drop_list)+['education']*len(education)+['tour']*len(restaurant)+['restaurant']*len(restaurant)+['culture']*len(culture)+\
                            ['recreation']*len(recreation)+['small_business']*len(small_business)+['hotel']*len(hotel)+['information']*len(information)+\
                            ['government']*len(government)+['medical']*len(medical)+['finance']*len(finance)))
#
new_type = poi_df['type'].replace(to_replace = replacement_dict)
poi_df['type_agg'] = new_type # new category of POIs

poi_df.to_pickle(intermediate_data_path+"POI_with_SA2_idx.pickle")
#endregion




#region 2. Create aggregate counts and entropy for SA2.
# output columns:
# poi_count, poi_entropy, poi_count_per_area, poi_entropy_per_area,
# poi_count_agg, poi_entropy_agg, poi_count_agg_per_area, poi_entropy_agg_per_area
with open(intermediate_data_path+"POI_with_SA2_idx.pickle", 'rb') as f:
    poi_df = pickle.load(f)
sa2_adelaide = gpd.read_file(intermediate_data_path+'shapefiles/sa2_adelaide.shp')

# remove the type_agg == drop
poi_agg_df = poi_df.loc[poi_df['type_agg'] != 'drop', :]

# create counts
count = poi_df.groupby(["SA2_MAIN16"],as_index=False).aggregate(["count"])[('geometry', 'count')]
count_agg = poi_agg_df.groupby(["SA2_MAIN16"],as_index=False).aggregate(["count"])[('geometry', 'count')] # miss one obs

# create entropy
def return_entropy(poi, count):
    '''
        return: entropy df
    '''

    split_count = poi.groupby(["SA2_MAIN16","type"],as_index=False).aggregate(["count"])
    entropy = {}
    for i, row in split_count.iterrows():
        sa_id, _type = i
        total_count = count.loc[sa_id]
        val = row[('geometry', 'count')] / total_count

        if sa_id not in entropy:
            entropy[sa_id] = (-val * np.log(val))
        else:
            entropy[sa_id] += (-val * np.log(val))

    entropy_df = pd.Series(entropy.values(), index=entropy.keys())
    return entropy_df

# compute two entropy values
entropy_df = return_entropy(poi_df, count)
entropy_agg_df = return_entropy(poi_agg_df, count_agg)

#
count.name = 'poi_count'
count_agg.name = 'poi_count_agg'
entropy_df.name = 'poi_entropy'
entropy_agg_df.name = 'poi_entropy_agg'

#
sa2_adelaide_merge=pd.merge(sa2_adelaide, count.to_frame(), left_on='SA2_MAIN16', right_index=True, how='outer')
sa2_adelaide_merge=sa2_adelaide_merge.merge(count_agg.to_frame(), left_on='SA2_MAIN16', right_index=True, how='outer')
sa2_adelaide_merge=sa2_adelaide_merge.merge(entropy_df.to_frame(), left_on='SA2_MAIN16', right_index=True, how='outer')
sa2_adelaide_merge=sa2_adelaide_merge.merge(entropy_agg_df.to_frame(), left_on='SA2_MAIN16', right_index=True, how='outer')

# nan and zeros exist.
print("Number of nan is: ", np.sum(sa2_adelaide_merge.isna()))
print("Number of zeros is: ", np.sum(sa2_adelaide_merge == 0))

# save the data
sa2_adelaide_merge.to_pickle(intermediate_data_path+'sa2_node_with_POI_counts_entropy.pickle')

#endregion













