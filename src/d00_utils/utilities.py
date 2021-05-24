import numpy as np
import pandas as pd
import geopandas as gpd
import momepy
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import linear_model
import statsmodels.api as sm


def compute_road_attributes(shp, road_shp):
    """
        Inputs:
            shp - a shp file
            road_shp - the shp file containing road information
        Outputs:
            One shapefile that incorporates the road class attributes
            One shapefile that incorporates the roads in only Adelaide area
    """
    
    print("=====Running union_road_land_shp=====")
        
    # create the centroids for roads
    road_centroid = road_shp.centroid
    
    # attach SA2 idx to road networks
    road_shp['SA2_loc'] = -1 # init as -1.

    for SA2_idx in range(shp.shape[0]):
        print(SA2_idx)
        
        # assign SA2_idx to the road network
        within_logic = road_centroid.within(shp.loc[SA2_idx, 'geometry'])
        road_shp.loc[within_logic, 'SA2_loc'] = SA2_idx
        
    # Use only the 'class' variable for now. 
    road_shp = road_shp[['class', 'geometry', 'SA2_loc']]
    road_shp_dummies = pd.get_dummies(road_shp)
    
    # aggregate the road attribute dummies for SA2.
    road_shp_dummies = road_shp_dummies.loc[road_shp_dummies['SA2_loc'] > -1]
    sa2_road_class_agg=road_shp_dummies.groupby(by='SA2_loc').sum()
    
    # augment road class variables to SA2_network.
    shp = shp.merge(sa2_road_class_agg, how='inner', left_index=True, right_index=True)
    
    # create road networks for only Adelaide
    road_shp_adelaide = road_shp.loc[road_shp['SA2_loc']>-1, :]

    print("=====DONE union_road_land_shp=====")
    
    return shp, road_shp_adelaide



def compute_intersection_attributes(shp_proj, road_proj):
    """
    Inputs:
        shp_proj - a shp projection merged with road info; the first output compute_road_attributes(shp, road_shp)
        road_proj - a shp projection that has only road information; the second output of compute_road_attributes(shp, road_shp)
    Outputs:
        degree_df - a df with SA2 code and its degree counts
    """

    count = {}
    
    for elt in road_proj["SA2_loc"]:
        if elt in count:
            count[elt] += 1
        else:
            count[elt] = 1
    SA_idxs = sorted((key,count[key]) for key in count)
    
    # create a dictionary to map sa_idx to road graphs
    sa_idx_to_graph = {}
    for sa_idx,c in SA_idxs:
        within = road_proj[road_proj["SA2_loc"]==sa_idx]
        graph = momepy.gdf_to_nx(within, approach='primal')
        sa_idx_to_graph[sa_idx] = graph

    # initialize the dataframe for intersections        
    degree_df = pd.DataFrame(columns=["SA2_MAIN16", "num_nodes", 
                                      "num_1degree", "num_2degree", 
                                      "num_3degree", "num_4degree", 
                                      "num_greater5degree"])

    # compute the degrees of roads for each SA_idx
    for sa_idx in sa_idx_to_graph:
        g = sa_idx_to_graph[sa_idx]
        degree = dict(nx.degree(g))
        nx.set_node_attributes(g, degree, 'degree')
        g = momepy.node_degree(g, name='degree')
        node_df, edge_df, sw = momepy.nx_to_gdf(g, points=True, lines=True, spatial_weights=True)

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
            
    return degree_df


def plot_sa2_node_attributes(node_shp, column_name, title_name, fig_name, save_path):
    '''
    plot the attributes for the SA2 nodes
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax) # for legend size
    cax = divider.append_axes("right", size="8%", pad=0.1) # for legend size
    node_shp.plot(facecolor='w', edgecolor='k', ax = ax)
    node_shp.plot(column = column_name, cmap='summer', legend=True, ax = ax, cax = cax) 
    # different cmaps: 'summer', 'OrRd'
    ax.set_title(title_name, fontsize=10)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(save_path+fig_name+'.png')
    plt.close()


def plot_sa2_edge_attributes(edge_shp, node_shp, column_name, title_name, fig_name, save_path):
    '''
    plot the attributes for the SA2 edges
    '''
    fig, ax = plt.subplots(figsize = (8, 8))
    node_shp.plot(facecolor='None', edgecolor='black', ax = ax, zorder = 10)
    node_shp.centroid.plot(ax = ax, facecolor = 'r', markersize = 5.0, zorder = 5)
    edge_shp.plot(column = column_name, cmap='turbo', legend=True, alpha = 0.5, ax = ax, zorder = 0)
    ax.set_title(title_name, fontsize=10)
    ax.set_axis_off()
    plt.tight_layout()    
    fig.savefig(save_path+fig_name+'.png')
    plt.close()


def plot_observed_predicted(df, y_name, x_name, model, save_path, picture_title, fig_name):
    '''
        inputs: dataframe, name of output, name of inputs, saved sm model.
        output: saved plot to save_path/fig_name.png
    '''
    y = np.log(df[y_name])
    X = np.log(df[x_name])
    X = sm.add_constant(X)
    pred_y = model.predict(X)
    fig, ax = plt.subplots(figsize = (4, 4))
    ax.scatter(y, pred_y, s = 0.5, color = 'g', marker='o')

    # rerun a y ~ pred_y regression for linear visualization.
    linear_visual_mod = sm.OLS(pred_y, sm.add_constant(y))
    linear_visual_res = linear_visual_mod.fit()
    y_list = np.array(np.linspace(np.min(y), np.max(y), 100))
    pred_y_list = linear_visual_res.predict(sm.add_constant(y_list))
    ax.plot(y_list, pred_y_list, linewidth = 2, color = 'r')

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_xlim(np.min(y_list) - .2* np.std(y_list), np.max(y_list) + .2* np.std(y_list))
    ax.set_ylim(np.min(pred_y) - .2* np.std(pred_y), np.max(pred_y) + .2* np.std(pred_y))
    ax.set_title(picture_title)
    ax.annotate("R2 = "+str(np.round(model.rsquared, 2)), xy=(.25, .85), xycoords='figure fraction')
    plt.tight_layout()
    fig.savefig(save_path+fig_name+".png")
    plt.close()


def latex_table(all_vars, models):
    # inputs:
    #      a full list of all the variables
    #      a list of statsmodel output models.
    # outputs:
    #      a latex table form.

    # create a base table for latex outputs
    errors = []
    for elt in all_vars:
        e = elt + "_err"
        errors.append(e)
    temp = []
    for i in range(len(all_vars)):
        temp.append(all_vars[i])
        if all_vars[i] not in {'Observations', 'R_squared', 'Adjusted_R_squared'}:
            temp.append(errors[i])

    table1 = pd.DataFrame(columns=temp)

    # create
    for variables, data, model in models:
        dict = {}
        for i, elt in enumerate(all_vars):
            if elt in model.params:
                i = list(model.params.keys()).index(elt)
                pval = model.pvalues[i]
                tag = ""
                if pval < 0.001:
                    tag = "***"
                elif pval < 0.01:
                    tag = "**"
                elif pval < 0.05:
                    tag = "*"
                dict[elt] = str(round(model.params[elt], 3)) + tag
                dict[elt + "_err"] = "(" + str(round(model.bse[elt], 3)) + ")"
            else:
                dict[elt] = ""
                if elt not in {'Observations', 'R_squared', 'Adjusted_R_squared'}:
                    dict[elt + "_err"] = ""
        dict["Observations"] = model.nobs
        dict["R_squared"] = round(model.rsquared, 3)
        dict["Adjusted_R_squared"] = round(model.rsquared_adj, 3)
        i = list(model.params.keys()).index("const")
        pval = model.pvalues[i]
        tag = ""
        if pval < 0.001:
            tag = "***"
        elif pval < 0.01:
            tag = "**"
        elif pval < 0.05:
            tag = "*"
        dict["Constant"] = str(round(model.params["const"], 3)) + tag
        dict["Constant_err"] = "(" + str(round(model.bse["const"], 3)) + ")"

        table1 = table1.append(dict, ignore_index=True)
    return table1


def post_lasso_estimate(y_name, x_attribute_names, alpha_value, df):
    '''
    :param y_name: name of output y
    :param x_attribute_names: a list of input x
    :param alpha_value: alpha value used for LASSO
    :param df: full dataframe
    :return: returns the input X and trained model
    '''
    y = np.log(df[y_name])
    X = np.log(df[x_attribute_names])
    X = sm.add_constant(X)
    mod = linear_model.Lasso(alpha=alpha_value) # 0.05.
    mod.fit(X, y)

    # choose only the sparse coefficients
    coeff_mask = np.abs(mod.coef_) > 0.00001
    coeff_mask = coeff_mask[1:]  # remove the first const
    x_attribute_names_sparse = x_attribute_names[coeff_mask]

    # choos
    y = np.log(df[y_name])
    X = np.log(df[x_attribute_names_sparse])
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()

    return X, x_attribute_names_sparse, res


def compute_econ_opportunity(metric_name, target_var_name, time_var_name, edge_df, attraction_param, friction_param, o_or_d):
    '''
    :param edge_df: edge data frame
    :param time_var_name: name of the travel duration variable
    :param target_var_name: resource variable
    :return: a dataframe with econ opportunity measure and SA2 idx
    '''
    if o_or_d == 'O':
        sa2_list = np.unique(edge_df['O'])
    elif o_or_d == 'D':
        sa2_list = np.unique(edge_df['D'])

    metric_list = []
    for sa2_idx in sa2_list:
        edge_df_sa2_specific = edge_df.loc[edge_df[o_or_d]==sa2_idx, :]
        metric=np.sum(edge_df_sa2_specific[target_var_name]**np.abs(attraction_param) / edge_df_sa2_specific[time_var_name]**np.abs(friction_param))
        metric_list.append(metric)

    #
    metric_df = pd.DataFrame({'sa2_code':sa2_list,
                              metric_name:metric_list})
    return metric_df






























