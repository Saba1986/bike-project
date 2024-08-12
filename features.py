#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Static + Strava feature list
strava_static_features = [
    'population_density_om', 'Bus Stops_hm', 'cycleway_track_all_hm', 'Primary_om', 
 'Footway_hm', 'log_stv_c_adb', 'Point Speed_hm', 'Distance to Grass', 'Residential_Road_om', 'point_slope_hm', 
 'Distance to Park Center', 'cycleway_lane_all_om', 'Student Access_hm', 'Distance to Commercial Area', 'tertiary_binary', 
 'Distance to Retail Center', 'Bicycle Parking_hm', 'uni_count_hm', 'min_dist_to_park', 'Secondary_om', 'School_om', 
 'Secondary_hm', 'BikeFac_hm', 'Residential_Area_hm', 'Median_HH_income_hm', 'residential_binary', 
 'Distance to Industrial Area', 'Tertiary_om', 'population_density_hm', 'cycleway_track_all_om', 'valid_months', 
 'Distance to Park', 'valid_year', 'Path_hm', 'Median Age_hm', 'Distance to Industrial Center', 'Cycleway_hm', 
 'footway_binary', 'adb', 'Median_HH_income_om', 'employment_density_om', 'BikeFac_om', 'Distance to Grass Center', 
 'reg_num', 'secondary_binary', 'region', 'log_stv_adb', 'Point Bridge_hm', 'Bike_Commuter_hm', 'Point Speed_om', 
 'University_om', 'lanes_om', 'cycleway_lane_binary', 'Number of jobs_hm', 'bridge_om', 'Bus Stops_om', 'Footway_om', 
 'college_hm', 'BikeFac_onstreet_hm', 'uni_count_om', 'point_slope_om', 'pct_at_least_college_education_hm', 
    'Distance to Retail Area', 'pct_at_least_college_education_om', 'Point Bridge_om', 'Water Area_om', 
    'Bicycle Parking_om', 'arterial_binary', 'Distance to Water Body', 'Median Age_om', 'sep_bikeway_binary', 
    'Cycleway_om', 'min_dist_to_university', 'Distance_to_Water_Body_mi', 'log_stv_nc_adb', 'edgeUID', 'college_om', 
    'avg_slope_hm', 'Path_om', 'HH_density_om', 'stv_c_adb', 'cycleway_lane_all_hm', 'Commercial Area_om', 
    'maj_uni_count_hm', 'valid_days_year', 'Distance to Commercial Area Center', 'Maj University_hm', 'Residential_Area_om', 
    'Student Access_om', 'Commercial Area_hm', 'HH_density_hm', 'lanes_hm', 'Intersection_Density_hm', 'University_hm', 
    'stv_adb', 'employment_density_hm', 'Number of jobs_om', 'min_dist_to_college', 'BikeFac_binary', 
    'avg_slope_om', 'Bike_Commuter_om', 'min_dist_to_maj_uni', 'Primary_hm', 'bridge_hm', 'stv_nc_adb', 'Water Area_hm', 
    'Intersection_Density_om', 'BikeFac_onstreet_om'
]


# Strava feature list
strava_features = ['valid_months', 'valid_days_year', 
                              'adb', 'valid_year', 'reg_num', 'stv_adb', 'stv_c_adb',
                              'stv_nc_adb', 'log_stv_adb', 'log_stv_c_adb', 'log_stv_nc_adb']


# static feature list
static_features = [
    'population_density_om', 'Bus Stops_hm', 'cycleway_track_all_hm', 'Primary_om', 
 'Footway_hm', 'Point Speed_hm', 'Distance to Grass', 'Residential_Road_om', 'point_slope_hm', 
 'Distance to Park Center', 'cycleway_lane_all_om', 'Student Access_hm', 'Distance to Commercial Area', 'tertiary_binary', 
 'Distance to Retail Center', 'Bicycle Parking_hm', 'uni_count_hm', 'min_dist_to_park', 'Secondary_om', 'School_om', 
 'Secondary_hm', 'BikeFac_hm', 'Residential_Area_hm', 'Median_HH_income_hm', 'residential_binary', 
 'Distance to Industrial Area', 'Tertiary_om', 'population_density_hm', 'cycleway_track_all_om', 
 'Distance to Park', 'Path_hm', 'Median Age_hm', 'Distance to Industrial Center', 'Cycleway_hm', 
 'footway_binary', 'Median_HH_income_om', 'employment_density_om', 'BikeFac_om', 'Distance to Grass Center', 
 'secondary_binary', 'region', 'Point Bridge_hm', 'Bike_Commuter_hm', 'Point Speed_om', 
 'University_om', 'lanes_om', 'cycleway_lane_binary', 'Number of jobs_hm', 'bridge_om', 'Bus Stops_om', 'Footway_om', 
 'college_hm', 'BikeFac_onstreet_hm', 'uni_count_om', 'point_slope_om', 'pct_at_least_college_education_hm', 
    'Distance to Retail Area', 'pct_at_least_college_education_om', 'Point Bridge_om', 'Water Area_om', 
    'Bicycle Parking_om', 'arterial_binary', 'Distance to Water Body', 'Median Age_om', 'sep_bikeway_binary', 
    'Cycleway_om', 'min_dist_to_university', 'Distance_to_Water_Body_mi', 'edgeUID', 'college_om', 
    'avg_slope_hm', 'Path_om', 'HH_density_om', 'cycleway_lane_all_hm', 'Commercial Area_om', 
    'maj_uni_count_hm', 'Distance to Commercial Area Center', 'Maj University_hm', 'Residential_Area_om', 
    'Student Access_om', 'Commercial Area_hm', 'HH_density_hm', 'lanes_hm', 'Intersection_Density_hm', 'University_hm', 
    'employment_density_hm', 'Number of jobs_om', 'min_dist_to_college', 'BikeFac_binary', 
    'avg_slope_om', 'Bike_Commuter_om', 'min_dist_to_maj_uni', 'Primary_hm', 'bridge_hm', 'Water Area_hm', 
    'Intersection_Density_om', 'BikeFac_onstreet_om'
]

