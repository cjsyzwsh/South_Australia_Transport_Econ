# evaluate the hypothetical scenario that highways have a better capacity.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import normalize
from sklearn import linear_model
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
model_path = os.path.join(os.getcwd(),'data/04_models/')
model_output_path = os.path.join(os.getcwd(),'data/05_model_outputs/')
report_path = os.path.join(os.getcwd(),'data/06_reporting/')

# read data
with open(processing_data_path+'node_df.pickle', 'rb') as f:
    node_df = pickle.load(f)
    node_df.reset_index()

with open(processing_data_path+'edge_consumption_df.pickle', 'rb') as f:
    edge_df = pickle.load(f) # use only edge_consumption_df for policy simulation
    edge_df = edge_df.reset_index() # reset

# read models
with open(model_path + 'model1_list_od_duration' + '.pickle', 'rb') as f:
    model1_list_od_duration = pickle.load(f)

with open(model_path + 'model2_list_consumption_amount_mcc_source' + '.pickle', 'rb') as f:
    model2_list_consumption_amount_mcc_source = pickle.load(f)

with open(model_path + 'model2_list_consumption_count_mcc_source' + '.pickle', 'rb') as f:
    model2_list_consumption_count_mcc_source = pickle.load(f)

with open(model_path + 'model2_list_flow_agents' + '.pickle', 'rb') as f:
    model2_list_flow_agents = pickle.load(f)

with open(model_path + 'model3_list_median_income_per_job_aud_persons' + '.pickle', 'rb') as f:
    model3_list_median_income_per_job_aud_persons = pickle.load(f)


###########################################################################################
# Forward pass for status quo and hypothetical case
###########################################################################################

def forward_simulation(edge_df, node_df, status_quo = True):
    # predict and change four variables in edge_df:
    #   od_duration, consumption_amount_mcc_source, consumption_count_mcc_source, flow_agents
    # predict five variables.
    #

    # model 1.
    # od_duration_pred is the output from the model 1.
    _, (y_name, x_name), model1 = model1_list_od_duration[-1]

    if status_quo == True:
        od_duration_pred = np.exp(model1.predict(sm.add_constant(np.log(edge_df[x_name])))) # remember the log and exponential transformation!
    else:
        model1.params['class_HWY'] = -0.48 # highway capacity increases by 20% compared to average
        od_duration_pred = np.exp(model1.predict(sm.add_constant(np.log(edge_df[x_name])))) # remember the log and exponential transformation!
    edge_df[y_name] = od_duration_pred # replace od_duration by predicted value

    # model 2.
    # three outputs:
    # model 2.1
    _, (y_name, x_name), model2_consumption_amount_mcc_source = model2_list_consumption_amount_mcc_source[-1]
    consumption_amount_mcc_source_pred = np.exp(model2_consumption_amount_mcc_source.predict(sm.add_constant(np.log(edge_df[x_name]))))
    edge_df[y_name] = consumption_amount_mcc_source_pred

    # model 2.2
    _, (y_name, x_name), model2_consumption_count_mcc_source = model2_list_consumption_count_mcc_source[-1]
    consumption_count_mcc_source_pred = np.exp(model2_consumption_count_mcc_source.predict(sm.add_constant(np.log(edge_df[x_name]))))
    edge_df[y_name] = consumption_count_mcc_source_pred

    # model 2.3
    _, (y_name, x_name), model2_flow_agents = model2_list_flow_agents[-1]
    flow_agents_pred = np.exp(model2_flow_agents.predict(sm.add_constant(np.log(edge_df[x_name]))))
    edge_df[y_name] = flow_agents_pred

    # model 3.
    _, (y_name, x_name), model3 = model3_list_median_income_per_job_aud_persons[-1]
    # change six columns in node_df: consumption_amount_mcc_source, consumption_count_mcc_source, flow_agents
    edge_df_relevant = edge_df[['O', 'D',
                                'consumption_amount_mcc_source',
                                'consumption_count_mcc_source',
                                'flow_agents']]

    sa2_flow_o = edge_df_relevant.groupby('O').aggregate(['sum']).reset_index()
    col_list = sa2_flow_o.columns
    col_new = [col[0]+'_o' for col in col_list]
    sa2_flow_o.columns = col_new

    sa2_flow_d = edge_df_relevant.groupby('D').aggregate(['sum']).reset_index()
    col_list = sa2_flow_d.columns
    col_new = [col[0]+'_d' for col in col_list]
    sa2_flow_d.columns = col_new

    node_df = node_df.merge(sa2_flow_o, left_on=['SA2_MAIN16'], right_on=['O_o'], how='left', suffixes=['_drop', ''])
    node_df = node_df.merge(sa2_flow_d, left_on=['SA2_MAIN16'], right_on=['D_d'], how='left', suffixes=['_drop', ''])

    # drop columns
    node_df = node_df.drop(columns=['D_o', 'O_d'])

    if status_quo == True:
        inc_pred = np.exp(model3.predict(sm.add_constant(np.log(node_df[x_name]))))
    else:
        model3.params['consumption_count_mcc_source_d'] = 0.0 # remove the insig parameter
        inc_pred = np.exp(model3.predict(sm.add_constant(np.log(node_df[x_name]))))

    return od_duration_pred, consumption_amount_mcc_source_pred, consumption_count_mcc_source_pred, flow_agents_pred, inc_pred

# 1. Use the status quo
status_quo = True
od_duration_pred, consumption_amount_mcc_source_pred, consumption_count_mcc_source_pred, flow_agents_pred, inc_pred = \
    forward_simulation(edge_df, node_df, status_quo)

# create a econ_opportunity_status_quo_dic
# format - metric_name: (target_var_name, time_var_name, attraction_param, friction_param, o_or_d)
econ_opportunity_input_dic = {'job_based_consumption_opportunity': ('job_density_1_o', 'od_duration', 0.047+0.064, 0.982, 'D'),
                                         'pop_based_consumption_opportunity': ('pop_density_o', 'od_duration', 0.055+0.078, 0.982, 'D'),
                                         'amenity_based_consumption_opportunity': ('poi_count_agg_density_o', 'od_duration', 0.037+0.141, 1.006, 'D'),
                                         'diversity_based_consumption_opportunity': ('poi_entropy_agg_density_o', 'od_duration', 0.084+0.066, 1.002, 'D')
                                         }

# status_quo_dic
econ_opportunity_status_quo_dic = {}
for metric_name in econ_opportunity_input_dic.keys():
    target_var_name, time_var_name, attraction_param, friction_param, o_or_d = econ_opportunity_input_dic[metric_name]
    metric_output = util.compute_econ_opportunity(metric_name, target_var_name, time_var_name, edge_df,
                                             attraction_param, friction_param, o_or_d)
    econ_opportunity_status_quo_dic[metric_name] = metric_output


# 2. Use the hypothetical case
status_quo = False
od_duration_pred_new, consumption_amount_mcc_source_pred_new, consumption_count_mcc_source_pred_new, flow_agents_pred_new, inc_pred_new = \
    forward_simulation(edge_df, node_df, status_quo)

# compute economic opportunity metric (hypothetical case)
econ_opportunity_hypo_dic = {}
for metric_name in econ_opportunity_input_dic.keys():
    target_var_name, time_var_name, attraction_param, friction_param, o_or_d = econ_opportunity_input_dic[metric_name]
    metric_output = util.compute_econ_opportunity(metric_name, target_var_name, time_var_name, edge_df,
                                             attraction_param, friction_param, o_or_d)
    econ_opportunity_hypo_dic[metric_name] = metric_output


# 3. Compare
print(np.sum(od_duration_pred_new - od_duration_pred))
print(np.sum(consumption_amount_mcc_source_pred_new - consumption_amount_mcc_source_pred))
print(np.sum(consumption_count_mcc_source_pred_new - consumption_count_mcc_source_pred))
print(np.sum(flow_agents_pred_new - flow_agents_pred))
print(np.sum(inc_pred_new - inc_pred)/np.sum(inc_pred))
print(np.sum(econ_opportunity_hypo_dic['job_based_consumption_opportunity']['job_based_consumption_opportunity'] - \
                econ_opportunity_status_quo_dic['job_based_consumption_opportunity']['job_based_consumption_opportunity']))
print(np.sum(econ_opportunity_hypo_dic['pop_based_consumption_opportunity']['pop_based_consumption_opportunity'] - \
                econ_opportunity_status_quo_dic['pop_based_consumption_opportunity']['pop_based_consumption_opportunity']))
print(np.sum(econ_opportunity_hypo_dic['amenity_based_consumption_opportunity']['amenity_based_consumption_opportunity'] - \
                econ_opportunity_status_quo_dic['amenity_based_consumption_opportunity']['amenity_based_consumption_opportunity']))
print(np.sum(econ_opportunity_hypo_dic['diversity_based_consumption_opportunity']['diversity_based_consumption_opportunity'] - \
                econ_opportunity_status_quo_dic['diversity_based_consumption_opportunity']['diversity_based_consumption_opportunity']))

# save simulation results.
# read node and edge.
with open(processing_data_path+'node_df.pickle', 'rb') as f:
    node_df = pickle.load(f)
    node_df.reset_index()

with open(processing_data_path+'edge_consumption_df.pickle', 'rb') as f:
    edge_df = pickle.load(f) # use only edge_consumption_df for policy simulation
    edge_df = edge_df.reset_index() # reset

edge_df['od_duration_save'] = od_duration_pred - od_duration_pred_new # reverse the order because new od_duration is smaller.
edge_df['consumption_amount_increase'] = consumption_amount_mcc_source_pred_new - consumption_amount_mcc_source_pred
edge_df['consumption_count_increase'] = consumption_count_mcc_source_pred_new - consumption_count_mcc_source_pred
edge_df['flow_agents_increase'] = flow_agents_pred_new - flow_agents_pred
node_df['income_increase'] = inc_pred_new - inc_pred
node_df['job_based_consumption_opportunity_increase'] = econ_opportunity_hypo_dic['job_based_consumption_opportunity']['job_based_consumption_opportunity'] - \
                            econ_opportunity_status_quo_dic['job_based_consumption_opportunity']['job_based_consumption_opportunity']
node_df['pop_based_consumption_opportunity_increase'] = econ_opportunity_hypo_dic['pop_based_consumption_opportunity']['pop_based_consumption_opportunity'] - \
                            econ_opportunity_status_quo_dic['pop_based_consumption_opportunity']['pop_based_consumption_opportunity']
node_df['amenity_based_consumption_opportunity_increase'] = econ_opportunity_hypo_dic['amenity_based_consumption_opportunity']['amenity_based_consumption_opportunity'] - \
                            econ_opportunity_status_quo_dic['amenity_based_consumption_opportunity']['amenity_based_consumption_opportunity']
node_df['diversity_based_consumption_opportunity_increase'] = econ_opportunity_hypo_dic['diversity_based_consumption_opportunity']['diversity_based_consumption_opportunity'] - \
                            econ_opportunity_status_quo_dic['diversity_based_consumption_opportunity']['diversity_based_consumption_opportunity']


# save
with open(model_output_path+'edge_df_policy_simulation.pickle', 'wb') as f:
    pickle.dump(edge_df, f)

with open(model_output_path+'node_df_policy_simulation.pickle', 'wb') as f:
    pickle.dump(node_df, f)



































