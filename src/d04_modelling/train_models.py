# three models to be trained.
# 1. travel time ~ travel distance + road attributes. [edge df]
# 2. mobility or consumption flow ~ origin attributes + destination attributes + travel time [edge df]
# 3. income ~ mobility flow + controls [node df]

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

# read data
with open(processing_data_path+'node_df.pickle', 'rb') as f:
    node_df = pickle.load(f)

with open(processing_data_path+'edge_consumption_df.pickle', 'rb') as f:
    edge_consumption_df = pickle.load(f)

with open(processing_data_path+'edge_flow_df.pickle', 'rb') as f:
    edge_flow_df = pickle.load(f) # more observations

# function

###########################################################################################
# model 1. infrastructure efficiency.
###########################################################################################
y_name = 'od_duration'
model_dic = {'Model 1': (y_name, ['od_distance']),
             'Model 2': (y_name, ['od_distance', 'num_roads', 'num_nodes']),
             'Model 3': (y_name, ['od_distance', 'num_roads', 'class_HWY', 'num_nodes', 'num_1degree'])}
edge_df = edge_flow_df
model_list = []
all_variables = ['Constant', 'od_distance', 'num_roads', 'class_HWY', 'num_nodes', 'num_1degree']

for model_idx in model_dic.keys():
    y_name, x_name = model_dic[model_idx]
    y = np.log(edge_df[y_name])
    X = np.log(edge_df[x_name])
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    model_list.append((model_idx, model_dic[model_idx], res))

# export the regression tables as latex
latex_table_results = util.latex_table(all_variables, model_list)
latex_table_results.index = model_dic.keys()
print(latex_table_results.T)

# save
with open(model_path+'model1_list_'+y_name+'.pickle', 'wb') as f:
    pickle.dump(model_list, f)
with open(model_output_path+'model1_table_'+y_name+'.txt', 'w') as f:
    f.writelines(latex_table_results.T.to_latex())



###########################################################################################
# model 2. human dynamics.
###########################################################################################
pd.set_option('display.max_columns', None) # change default columns to max.

# y_name = 'flow_agents'
# edge_df = edge_flow_df
# alpha_value = 0.08

y_name = 'consumption_count_mcc_source'
edge_df = edge_consumption_df
alpha_value = 0.03

# y_name = 'consumption_amount_mcc_source'
# edge_df = edge_consumption_df
# alpha_value = 0.02

model_dic = {'Model 1': (y_name, ['od_duration']),
             'Model 2': (y_name, ['od_duration', 'pop_density_o', 'pop_density_d']),
             'Model 3': (y_name, ['od_duration', 'job_density_1_o', 'job_density_1_d']),
             'Model 4': (y_name, ['od_duration', 'poi_count_agg_density_o', 'poi_count_agg_density_d']),
             'Model 5': (y_name, ['od_duration', 'poi_entropy_agg_density_o', 'poi_entropy_agg_density_d'])}
model_list = []
all_variables = ['Constant', 'od_duration', 'pop_density_o', 'pop_density_d', 'job_density_1_o', 'job_density_1_d',
                 'poi_count_agg_density_o', 'poi_count_agg_density_d', 'poi_entropy_agg_density_o', 'poi_entropy_agg_density_d']

# Train the first five models
for model_idx in model_dic.keys():
    y_name, x_name = model_dic[model_idx]
    y = np.log(edge_df[y_name])
    X = np.log(edge_df[x_name])
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    model_list.append((model_idx, model_dic[model_idx], res))

# Train the last model
x_attribute_names = np.array(['od_duration',
                              'poi_count_agg_density_o', 'poi_count_agg_density_d',
                              'poi_entropy_density_o', 'poi_entropy_density_d',
                              'poi_entropy_agg_density_o', 'poi_entropy_agg_density_d',
                              'pop_density_o', 'pop_density_d',
                              'job_density_1_o', 'job_density_1_d'
                              ])

# train the post lasso model
X, x_attribute_names_sparse, res = util.post_lasso_estimate(y_name, x_attribute_names, alpha_value, edge_df)
model_list.append(('Post LASSO', (y_name, x_attribute_names_sparse), res))

# latex outputs
latex_table_results = util.latex_table(all_variables, model_list)
latex_table_results.index = list(model_dic.keys()) + ['Post LASSO']
print(latex_table_results.T)
# latex_table_results.T.to_latex()

# save
with open(model_path+'model2_list_'+ y_name +'.pickle', 'wb') as f:
    pickle.dump(model_list, f)
with open(model_output_path+'model2_table_'+ y_name +'.txt', 'w') as f:
    f.writelines(latex_table_results.T.to_latex())



###########################################################################################
# model 3. economic outcomes
###########################################################################################
# all potential ys: median_income_per_job_aud_persons, unemployment_rate, poverty_rate_1, poverty_rate_2, median_inc, gini.
y_name = 'median_income_per_job_aud_persons'
# y_name = 'num_jobs_000_persons'
alpha_value = 0.01

model_dic = {'Model 1': (y_name, ['flow_agents_o', 'flow_agents_d']),
             'Model 2': (y_name, ['consumption_amount_mcc_source_o', 'consumption_amount_mcc_source_d']),
             'Model 3': (y_name, ['consumption_count_mcc_source_o', 'consumption_count_mcc_source_d']),
             'Model 4': (y_name, ['flow_agents_o', 'flow_agents_d',
                                  'consumption_amount_mcc_source_o', 'consumption_amount_mcc_source_d',
                                  'consumption_count_mcc_source_o', 'consumption_count_mcc_source_d'
                                  ]),
             'Model 5': (y_name, ['flow_agents_o', 'flow_agents_d',
                                  'consumption_amount_mcc_source_o', 'consumption_amount_mcc_source_d',
                                  'consumption_count_mcc_source_o', 'consumption_count_mcc_source_d',
                                  'total_pop','avg_age', 'male_percent', 'bachelor_degree_percent', 'master_degree_percent'
                                  ])}
model_list = []
all_variables = ['Constant', 'flow_agents_o', 'flow_agents_d',
                 'consumption_amount_mcc_source_o', 'consumption_amount_mcc_source_d',
                 'consumption_count_mcc_source_o', 'consumption_count_mcc_source_d',
                 'total_pop','avg_age', 'male_percent', 'bachelor_degree_percent', 'master_degree_percent']

# Train the first five models
for model_idx in model_dic.keys():
    y_name, x_name = model_dic[model_idx]
    y = np.log(node_df[y_name])
    X = np.log(node_df[x_name])
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    model_list.append((model_idx, model_dic[model_idx], res))

# Train the last model
x_attribute_names = np.array(['flow_agents_o', 'flow_agents_d',
                              'consumption_amount_mcc_source_o', 'consumption_amount_mcc_source_d',
                              'consumption_count_mcc_source_o', 'consumption_count_mcc_source_d',
                              'total_pop','avg_age', 'male_percent', 'bachelor_degree_percent', 'master_degree_percent'
                              ]) # or pop_density?

# train the post lasso model
X, x_attribute_names_sparse, res = util.post_lasso_estimate(y_name, x_attribute_names, alpha_value, node_df)
model_list.append(('Post LASSO', (y_name, x_attribute_names_sparse), res))

# latex outputs
latex_table_results = util.latex_table(all_variables, model_list)
latex_table_results.index = list(model_dic.keys()) + ['Post LASSO']
print(latex_table_results.T)

# save
with open(model_path+'model3_list_'+ y_name +'.pickle', 'wb') as f:
    pickle.dump(model_list, f)
with open(model_output_path+'model3_table_'+ y_name +'.txt', 'w') as f:
    f.writelines(latex_table_results.T.to_latex())


###########################################################################################
# model 4. Scaling diagnosis
###########################################################################################
X = np.log(node_df['pop_density'])
y_list = ['median_income_per_job_aud_persons',
          'flow_agents_o',
          'flow_agents_d',
          'consumption_count_mcc_source_o',
          'consumption_count_mcc_source_d',
          'consumption_amount_mcc_source_o',
          'consumption_amount_mcc_source_d',
          'poi_count_agg',
          'poi_count_agg_density',
          'poi_entropy_agg',
          'poi_entropy_agg_density',
          'num_roads',
          'num_nodes'
          ]

model_list = []

# param_dic = {}
for y_name in y_list:
    # print(y_name)
    y = np.log(node_df[y_name])
    mod = sm.OLS(y, X)
    res = mod.fit()
    model_list.append(('_', (y_name, 'pop_density'), res))

with open(model_path+'model4_list_scaling_''.pickle', 'wb') as f:
    pickle.dump(model_list, f)



















