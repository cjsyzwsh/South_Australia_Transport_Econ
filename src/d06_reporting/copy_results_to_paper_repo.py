# this script moves the useful outputs to the paper repository.

import pandas as pd
import shutil
import os
import sys

# from path
model_output_path = os.path.join(os.getcwd(),'data/05_model_outputs/')
report_path = os.path.join(os.getcwd(),'data/06_reporting/')

# to overleaf file
overleaf_path = '/Users/shenhaowang/Dropbox (MIT)/Apps/Overleaf/Gravity model for econ and transport networks/'

# from & to file
file_dic = {}

# tables
file_dic[model_output_path+'model1_table_od_duration.txt'] = overleaf_path+'tables/'+'model1_table_od_duration.txt'
file_dic[model_output_path+'model2_table_consumption_amount_mcc_source.txt'] = overleaf_path+'tables/'+'model2_table_consumption_amount_mcc_source.txt'
file_dic[model_output_path+'model2_table_consumption_count_mcc_source.txt'] = overleaf_path+'tables/'+'model2_table_consumption_count_mcc_source.txt'
file_dic[model_output_path+'model2_table_flow_agents.txt'] = overleaf_path+'tables/'+'model2_table_flow_agents.txt'
file_dic[model_output_path+'model3_table_median_income_per_job_aud_persons.txt'] = overleaf_path+'tables/'+'model3_table_median_income_per_job_aud_persons.txt'

# figs
#
file_dic[report_path+'local_environment/'+'Adelaide_area.png'] = overleaf_path+'figs/'+'Adelaide_area.png'
#
file_dic[report_path+'node_visual/'+'node_socioecon_median_income.png'] = overleaf_path+'figs/'+'node_socioecon_median_income.png'
file_dic[report_path+'node_visual/'+'node_socioecon_pop_density.png'] = overleaf_path+'figs/'+'node_socioecon_pop_density.png'
file_dic[report_path+'node_visual/'+'node_poi_count_agg_density.png'] = overleaf_path+'figs/'+'node_poi_count_agg_density.png'
file_dic[report_path+'node_visual/'+'node_poi_entropy_agg_density.png'] = overleaf_path+'figs/'+'node_poi_entropy_agg_density.png'
#
file_dic[report_path+'edge_visual/'+'edge_consumption_amount_mcc_source.png'] = overleaf_path+'figs/'+'edge_consumption_amount_mcc_source.png'
file_dic[report_path+'edge_visual/'+'edge_consumption_count_mcc_source.png'] = overleaf_path+'figs/'+'edge_consumption_count_mcc_source.png'
file_dic[report_path+'edge_visual/'+'edge_flow_agents.png'] = overleaf_path+'figs/'+'edge_flow_agents.png'
file_dic[report_path+'edge_visual/'+'edge_consumption_amount_mcc_source_sparse.png'] = overleaf_path+'figs/'+'edge_consumption_amount_mcc_source_sparse.png'
file_dic[report_path+'edge_visual/'+'edge_consumption_count_mcc_source_sparse.png'] = overleaf_path+'figs/'+'edge_consumption_count_mcc_source_sparse.png'
file_dic[report_path+'edge_visual/'+'edge_flow_agents_sparse.png'] = overleaf_path+'figs/'+'edge_flow_agents_sparse.png'
#
file_dic[report_path+'model_visual_pred_actual/'+'pred_consumption_amount.png'] = overleaf_path+'figs/'+'pred_consumption_amount.png'
file_dic[report_path+'model_visual_pred_actual/'+'pred_consumption_count.png'] = overleaf_path+'figs/'+'pred_consumption_count.png'
file_dic[report_path+'model_visual_pred_actual/'+'pred_flow.png'] = overleaf_path+'figs/'+'pred_flow.png'
file_dic[report_path+'model_visual_pred_actual/'+'pred_inc.png'] = overleaf_path+'figs/'+'pred_inc.png'
file_dic[report_path+'model_visual_pred_actual/'+'pred_travel_time.png'] = overleaf_path+'figs/'+'pred_travel_time.png'
#
file_dic[report_path+'policy_simulation/'+'simulation_consumption_amount_increase.png'] = overleaf_path + 'figs/'+'simulation_consumption_amount_increase.png'
file_dic[report_path+'policy_simulation/'+'simulation_consumption_count_increase.png'] = overleaf_path + 'figs/'+'simulation_consumption_count_increase.png'
file_dic[report_path+'policy_simulation/'+'simulation_flow_agents_increase.png'] = overleaf_path + 'figs/'+'simulation_flow_agents_increase.png'
file_dic[report_path+'policy_simulation/'+'simulation_od_duration_save.png'] = overleaf_path + 'figs/'+'simulation_od_duration_save.png'
file_dic[report_path+'policy_simulation/'+'simulation_income_increase.png'] = overleaf_path + 'figs/'+'simulation_income_increase.png'
file_dic[report_path+'policy_simulation/'+'simulation_income_increase_ratio.png'] = overleaf_path + 'figs/'+'simulation_income_increase_ratio.png'
file_dic[report_path+'policy_simulation/'+'simulation_diversity_based_consumption_opp.png'] = overleaf_path + 'figs/'+'simulation_diversity_based_consumption_opp.png'
file_dic[report_path+'policy_simulation/'+'simulation_amenity_based_consumption_opp.png'] = overleaf_path + 'figs/'+'simulation_amenity_based_consumption_opp.png'


# copy and paste
for key_ in file_dic.keys():
    from_file = key_
    target_file = file_dic[key_]
    shutil.copy2(from_file, target_file)


















