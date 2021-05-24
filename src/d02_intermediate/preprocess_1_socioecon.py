# preprocessing the socioeconomic variables
# inputs: five raw socio-demographcics data.
# outputs: one combined socio-econ geopandas data frame.

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

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


# # raw data
# mount_path = "/Users/shenhaowang/Dropbox (MIT)/project_econ_opportunity_south_Australia"


#region 1. read and edit job and income data frames
jobs_all = pd.read_csv(raw_data_path + "SA2_Jobs_All_Jobs_and_Income.csv")
jobs_industries = pd.read_csv(raw_data_path + "SA2_Jobs_In_Australia_Employee_Jobs_and_Income.csv")

# change column names for the two jobs data frame.
new_idx = []
for col in jobs_industries.columns:
    if col[0] == ' ':
        new_idx.append(col[1:])
    else:
        new_idx.append(col)

jobs_industries.columns = new_idx

new_idx = []
for col in jobs_all.columns:
    if col[0] == ' ':
        new_idx.append(col[1:])
    else:
        new_idx.append(col)

jobs_all.columns = new_idx

# change types of job data frames
jobs_all['sa2_code16'] = jobs_all['sa2_code16'].astype('str')
jobs_industries['sa2_code16'] = jobs_industries['sa2_code16'].astype('str')
# useful variables: all
#endregion


#region 2. read and edit age dataframe
age_df = pd.read_csv(raw_data_path + "data_age.csv")
# choose one section of age df
age_df = age_df[[' sa2_main16', 'p_tot_75_84_yrs', ' p_tot_35_44_yrs', ' p_tot_45_54_yrs', ' p_tot_25_34_yrs', ' p_tot_85ov',
     ' p_tot_65_74_yrs', ' p_tot_20_24_yrs', ' p_tot_15_19_yrs', ' p_tot_55_64_yrs', ' p_tot_tot']]

for col in ['p_tot_75_84_yrs', ' p_tot_35_44_yrs', ' p_tot_45_54_yrs', ' p_tot_25_34_yrs', ' p_tot_85ov',
            ' p_tot_65_74_yrs', ' p_tot_20_24_yrs', ' p_tot_15_19_yrs', ' p_tot_55_64_yrs']:
    age_df[col] = age_df[col] / age_df[" p_tot_tot"]  # compute percentage

avg_med_age = []
for idx, row in age_df.iterrows():
    avg = 0
    for col in ['p_tot_75_84_yrs', ' p_tot_35_44_yrs',
                ' p_tot_45_54_yrs', ' p_tot_25_34_yrs', ' p_tot_85ov',
                ' p_tot_65_74_yrs', ' p_tot_20_24_yrs', ' p_tot_15_19_yrs',
                ' p_tot_55_64_yrs']:
        if col != " p_tot_85ov":
            temp = col.split("_")
            x = int(temp[2])
            y = int(temp[3])
            med = (x + y) / 2
            avg += (row[col] * med)
        else:
            avg += (row[col] * 85)
        # avg = avg/row[' p_tot_tot']
    avg_med_age.append(avg)

# create average age column
age_df["avg_age"] = avg_med_age

# edit sa2_main16 type
age_df[' sa2_main16'] = age_df[' sa2_main16'].astype('str')

# rename the columns
rename_dic = {'p_tot_75_84_yrs':'percent_75_84_yrs',
              ' p_tot_35_44_yrs':'percent_35_44_yrs',
              ' p_tot_45_54_yrs':'percent_45_54_yrs',
              ' p_tot_25_34_yrs':'percent_25_34_yrs',
              ' p_tot_85ov':'percent_85ov_yrs',
              ' p_tot_65_74_yrs':'percent_65_74_yrs',
              ' p_tot_20_24_yrs':'percent_20_24_yrs',
              ' p_tot_15_19_yrs':'percent_15_19_yrs',
              ' p_tot_55_64_yrs':'percent_55_64_yrs',
              ' p_tot_tot':'total_pop',
              ' sa2_main16':'sa2_main16'}


age_df.rename(columns=rename_dic, inplace=True)
# note: average age = NaN or zero exists, because many zones don't have population..
# useful variables: all.
#endregion


#region 3. read and edit gender and education dataframe
gender_educ_df = pd.read_csv(raw_data_path + "data_gender_educ.csv")
all_educ_df = pd.read_csv(raw_data_path + "data_all_educ.csv")

#
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

gender_educ_df["male_percent"] = gender_educ_df[" m_tot_tot"]/gender_educ_df[" gender_tot_tot"]
gender_educ_df["female_percent"] = gender_educ_df[" f_tot_tot"]/gender_educ_df[" gender_tot_tot"]

# edit sa2_main var
helper = [str(elt) for elt in gender_educ_df[" sa2_main16"].values]
gender_educ_df["SA2_MAIN16"] = helper

#
all_educ_df['bachelor_degree_percent'] = all_educ_df[' p_b_deg_tot']/all_educ_df[' p_tot_tot']
all_educ_df['master_degree_percent'] = all_educ_df[' p_grad_dip_cer_tot']/all_educ_df[' p_tot_tot']

#
all_educ_df.rename(columns={' sa2_main16':'sa2_main16'},inplace=True)

#
all_educ_df['sa2_main16']=all_educ_df['sa2_main16'].astype('str')

#
gender_educ_df = gender_educ_df[['SA2_MAIN16', 'male_percent', 'female_percent']]
all_educ_df = all_educ_df[['sa2_main16', 'bachelor_degree_percent', 'master_degree_percent']]
#endregion


#region 4. read and edit indigenous social variables
indigenous_social_df = pd.read_csv(raw_data_path + "data_indigenous.csv")

# replace names
rename_dic = {'perc_indig_age_0_14':'perc_indig_age_0_14',
              ' perc_indig_hsld_equiv_inc_less_than_300':'perc_indig_hsld_equiv_inc_less_than_300',
              ' perc_indig_rent_oth_dwl':'perc_indig_rent_oth_dwl',
              ' perc_indig_no_vehicle_in_hsld':'perc_indig_no_vehicle_in_hsld',
              ' perc_indig_age_65_over':'perc_indig_age_65_over',
              ' perc_indig_rent_priv_dwl':'perc_indig_rent_priv_dwl',
              ' perc_indig_rent_pub_dwl':'perc_indig_rent_pub_dwl',
              ' perc_indig_f':'perc_indig_f',
              ' perc_indig_owned_outright_dwl':'perc_indig_owned_outright_dwl',
              ' perc_indig_age_35_64':'perc_indig_age_35_64',
              ' perc_indig_age_15plus_edu_degree_diploma_certificate':'perc_indig_age_15plus_edu_degree_diploma_certificate',
              ' perc_indig_hsld_equiv_inc_1000_1500':'perc_indig_hsld_equiv_inc_1000_1500',
              ' perc_indig_1_or_more_vehicle_in_hsld':'perc_indig_1_or_more_vehicle_in_hsld',
              ' perc_indig_hsld_equiv_inc_above_1500':'perc_indig_hsld_equiv_inc_above_1500',
              ' perc_indig_hsld_equiv_inc_300_1000':'perc_indig_hsld_equiv_inc_300_1000',
              ' sa2_code16':'sa2_code16',
              ' perc_indig_m':'perc_indig_m',
              ' perc_indig_age_15_34':'perc_indig_age_15_34',
              ' perc_indig_age_15plus_edu_none':'perc_indig_age_15plus_edu_none'}

indigenous_social_df.rename(columns=rename_dic,inplace=True)

indigenous_social_df['sa2_code16']=indigenous_social_df['sa2_code16'].astype('str')

# print(indigenous_social_df.shape)
#endregion


#region 5. Other socio economic variables
econ_df = pd.read_csv(raw_data_path+'social_econ_indicators.csv')
unemployment_rate_df = pd.read_csv(raw_data_path+'data_unemployment_rate.csv')

#
rename_dic = {'pov_rt_exc_hc_syn':'poverty_rate_1',
              ' housestrs_syn': 'hh_finance_stress',
              ' equivinc_median_syn':'equivinc_median_syn',
              ' pov_rt_syn':'poverty_rate_2',
              ' inc_median_syn':'median_inc',
              ' gini_syn':'gini',
              ' sa2_code16': 'sa2_code16'}
econ_df.rename(columns=rename_dic,inplace=True)
econ_df['sa2_code16']=econ_df['sa2_code16'].astype('str')


#
rename_dic = {'unemployment_rate':'unemployment_rate',
              ' sa2_code16': 'sa2_code16'}
unemployment_rate_df.rename(columns=rename_dic,inplace=True)
unemployment_rate_df['sa2_code16']=unemployment_rate_df['sa2_code16'].astype('str')

#endregion




#region 6. merge all socio-economic variables
# jobs_all, jobs_industries, age_df, gender_educ_df, all_educ_df, indigenous_social_df
# print(jobs_all.columns) # sa2_code16
# print(jobs_all.shape)
# print(jobs_industries.columns) # sa2_code16
# print(jobs_industries.shape)
# print(age_df.columns) # sa2_main16
# print(age_df.shape)
# print(gender_educ_df.columns) # SA2_MAIN16
# print(gender_educ_df.shape)
# print(all_educ_df.columns) # sa2_main16
# print(all_educ_df.shape)
# print(indigenous_social_df.columns) # sa2_code16
# print(indigenous_social_df.shape)

socio_econ_df = jobs_all.merge(jobs_industries, on='sa2_code16', suffixes=("","_y"))
socio_econ_df = socio_econ_df.merge(age_df, left_on='sa2_code16', right_on='sa2_main16')
socio_econ_df = socio_econ_df.merge(gender_educ_df, left_on='sa2_code16', right_on='SA2_MAIN16')
socio_econ_df = socio_econ_df.merge(all_educ_df, left_on='sa2_code16', right_on='sa2_main16')
socio_econ_df = socio_econ_df.merge(indigenous_social_df, on='sa2_code16', suffixes=("","_z"))
socio_econ_df = socio_econ_df.merge(econ_df, on='sa2_code16', suffixes=("","_drop"))
socio_econ_df = socio_econ_df.merge(unemployment_rate_df, on='sa2_code16', suffixes=("","_drop"))

print(socio_econ_df.shape)
print(socio_econ_df.columns)
#endregion


# save files
socio_econ_df.to_pickle(intermediate_data_path+'sa2_node_with_socio_econ_df.pickle') # Pycharm code
# socio_econ_df.to_pickle('../data/socio_econ_df.pickle') # command line code.






