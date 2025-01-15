# -*- coding: utf-8 -*-

# This code takes the 8:00 scenario as an example to demonstrate the optimization process, the SAA solution method, and the approach for evaluating the quality of the approximate solutions.
# The obtained rebalancing decisions are represented by the "dispatch_vehicle_everymin" variable, while the evaluation metrics for the approximate solutions are captured by the "gap_record", "gap_percent_record", "var_record", and "std_record" variables.

import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import sys
sys.path.append(r'***') # Please change "***" to the file path where the "Vehicle_dispatch_optimization_model.py" is located
import multiprocessing as mp
from Vehicle_dispatch_optimization_model import dispatch_optimization




# Estimated number of vehicles per minute in each region within "Tveh"
# This calculation has already accounted for the current number of passengers and vehicles, the predicted passenger numbers within "Tshort" (assuming passengers arrive uniformly during "Tshort"), and the number of vehicles arriving per minute within "Tveh". It also projects the remaining number of passengers in each region for every minute over the next 10 minutes, factoring in the matching of passengers and vehicles per minute during this period.
station_vehicle_redundant = np.load('***/station_vehicle_redundant_demo_18pm.npy') 

# Estimated number of passengers per minute in each region within "Tshort"
# This calculation has already accounted for the current number of passengers and vehicles, the predicted passenger numbers within "Tshort" (assuming passengers arrive uniformly during "Tshort"), and the number of vehicles arriving per minute within "Tveh". It also projects the remaining number of passengers in each region for every minute over the next 10 minutes, factoring in the matching of passengers and vehicles per minute during this period.
station_passenger_redundant = np.load('***/station_passenger_redundant_demo_18pm.npy') 

# Predicted passenger demand in each region within "Ttrend"
station_passenger_arrive_30min = np.load('***/station_passenger_arrive_30min_demo_18pm.npy') 

# Predicted passenger demand in all regions within "Ttrend"
station_passenger_arrive_30min_allzone = np.load('***/station_passenger_arrive_30min_allzone_demo_18pm.npy') 

# The reachable regions for each area within the rebalancing time window "Treb", simulated on a per-minute basis
taxi_zones_neighbor_10min_extension = np.load('***/taxi_zones_neighbor_10min_extension.npy') 

# The actual road travel distances between the centroids of each region
taxi_zones_centroid_route_distance = pd.read_csv(r'***/taxi_zones_centroid_route_distance.csv').values 



# Parameters
opt_num = 1080             # The optimization period corresponding to 18:00
sample_num_M = 20         # M groups of independent identically distributed samples
sample_num_N = 40         # N random samples in each group
sample_num_N_2 = 50000    # N_2 additional samples to get an estimated upper bound of the optimal value
i_iter = list(range(263)) # A total of 263 regions

# Optimization parameters
alpha1 = 10   # Weight for supply-demand gap
alpha2 = 1    # Weight for rebalancing distance
alpha3 = 100  # Weight for penalty intensity
gamma = 1/2   # Passenger demand satisfaction parameter
multi_processes = 10  # Set for parallel computation

# Generate random scenes
# Create sets of scenarios with uncertain prediction error distribution, using truncated normal distribution to avoid samples smaller than 0
# Generate M groups with N samples

sto_set = np.zeros((sample_num_M,sample_num_N,264))
for m in range(sample_num_M):
    for j in range(263): 
        # Samples are generated based on the fitting results of error distributions segmented by predicted passenger demand
        if station_passenger_arrive_30min[opt_num,j] < 5: 
            sto_set[m,:,j] = stats.truncnorm((0 - 0.898704) / 1.46148 ,10000000000, loc=0.898704, scale=1.46148).rvs(sample_num_N)
        if station_passenger_arrive_30min[opt_num,j] >= 5 and station_passenger_arrive_30min[opt_num,j] < 20:
            sto_set[m,:,j] = stats.truncnorm((-5 - 0.870471) / 4.24424,10000000000, loc=0.870471, scale=4.24424).rvs(sample_num_N)
        if station_passenger_arrive_30min[opt_num,j] >= 20 and station_passenger_arrive_30min[opt_num,j] < 40:
            sto_set[m,:,j] = stats.truncnorm((-20 - 1.56748) / 7.10412,10000000000, loc=1.56748, scale=7.10412).rvs(sample_num_N)
        if station_passenger_arrive_30min[opt_num,j] >= 40 and station_passenger_arrive_30min[opt_num,j] < 60:
            sto_set[m,:,j] = stats.truncnorm((-40 - 2.04633) / 10.2083,10000000000, loc=2.04633, scale=10.2083).rvs(sample_num_N)
        if station_passenger_arrive_30min[opt_num,j] >= 60 and station_passenger_arrive_30min[opt_num,j] < 80:
            sto_set[m,:,j] = stats.truncnorm((-60 - 2.22749) / 13.0313,10000000000, loc=2.22749, scale=13.0313).rvs(sample_num_N)
        if station_passenger_arrive_30min[opt_num,j] >= 80 and station_passenger_arrive_30min[opt_num,j] < 100:
            sto_set[m,:,j] = stats.truncnorm((-80 - 3.31231) / 16.2504,10000000000, loc=3.31231, scale=16.2504).rvs(sample_num_N)
        if station_passenger_arrive_30min[opt_num,j] >= 100 and station_passenger_arrive_30min[opt_num,j] < 400:            
            sto_set[m,:,j] = stats.truncnorm((-100- 6.87418) / 25.9947,10000000000, loc=6.87418, scale=25.9947).rvs(sample_num_N)
    for k in range(sample_num_N):
        sto_set[m,k,263] = sum(sto_set[m,k,0:263]) # The sum of uncertainty samples across all regions

# Record the optimal solution for each optimization group
result_test_x_all = np.zeros((sample_num_M,263,263))
result_test_y_all = np.zeros((sample_num_M,sample_num_N,263))
result_test_beta_all = np.zeros((sample_num_M,263))
result_obj_all = np.zeros(sample_num_M)

# Parallel optimization
for multi_iter in range(int(sample_num_M / multi_processes)):
    
    if __name__ == '__main__':   

        pool = mp.Pool(processes=multi_processes)
        result_all = []    
        for sample_iter in range(multi_processes):
            
            results1 = pool.apply_async(dispatch_optimization, args=(sample_num_N, multi_iter*multi_processes+ sample_iter, opt_num, station_vehicle_redundant, station_passenger_redundant, station_passenger_arrive_30min_allzone, station_passenger_arrive_30min,  sto_set, taxi_zones_neighbor_10min_extension, taxi_zones_centroid_route_distance,))
            result_all.append(results1)      
         
        pool.close()
        pool.join()
    
        print("Sub-process(es) done.")
        
        # Record the optimal solution and optimal value for each optimization group
        for res_iter in range(multi_processes):
            
            result_test_x_all[multi_iter*multi_processes+ res_iter,:,:] = result_all[res_iter].get()[0]
            result_test_beta_all[multi_iter*multi_processes+ res_iter,:] = result_all[res_iter].get()[1]
            result_test_y_all[multi_iter*multi_processes+ res_iter,:,:] = result_all[res_iter].get()[2]
            
            result_obj_all[multi_iter*multi_processes+ res_iter] =  (alpha1 * 1 / sample_num_N * (sum(result_test_y_all[multi_iter * multi_processes + res_iter,sto_scene,n] for n in range(263) for sto_scene in range(sample_num_N)))+ 
                                                                    alpha1 * 1 / sample_num_N * (sum(result_test_y_all[multi_iter * multi_processes + res_iter,sto_scene,0] for sto_scene in range(sample_num_N)))+ #Since Newark Airport is located at a considerable distance, 10 km can be added to the original dispatching distance of Newark Airport
                                                                    alpha2 * sum(taxi_zones_centroid_route_distance[m, n]/1000 * result_test_x_all[multi_iter * multi_processes + res_iter,m, n] for m in range(263) for n in range(263)) + 
                                                                    alpha3 * sum(result_test_beta_all[multi_iter * multi_processes + res_iter,n] for n in range(263)))
            

# Estimate the lower bound of the optimal objective value of the original problem
result_obj_lowerbound = np.average(result_obj_all)
# Estimate the variance
result_obj_var_1 = 1/(sample_num_M * (sample_num_M-1)) * (sum((result_obj_all[:] - result_obj_lowerbound)**2))
        
             
# Estimate the lower bound of the optimal objective value of the original problem      
# Generate N' samples
sto_set_2 = np.zeros((sample_num_N_2,264))
        
for j in range(263):
    # Samples are generated based on the fitting results of error distributions segmented by predicted passenger demand
    if station_passenger_arrive_30min[opt_num,j] < 5: 
        sto_set_2[:,j] = stats.truncnorm((0 - 0.898704) / 1.46148 ,10000000000, loc=0.898704, scale=1.46148).rvs(sample_num_N_2)
    if station_passenger_arrive_30min[opt_num,j] >= 5 and station_passenger_arrive_30min[opt_num,j] < 20:
        sto_set_2[:,j] = stats.truncnorm((-5 - 0.870471) / 4.24424,10000000000, loc=0.870471, scale=4.24424).rvs(sample_num_N_2)
    if station_passenger_arrive_30min[opt_num,j] >= 20 and station_passenger_arrive_30min[opt_num,j] < 40:
        sto_set_2[:,j] = stats.truncnorm((-20 - 1.56748) / 7.10412,10000000000, loc=1.56748, scale=7.10412).rvs(sample_num_N_2)
    if station_passenger_arrive_30min[opt_num,j] >= 40 and station_passenger_arrive_30min[opt_num,j] < 60:
        sto_set_2[:,j] = stats.truncnorm((-40 - 2.04633) / 10.2083,10000000000, loc=2.04633, scale=10.2083).rvs(sample_num_N_2)
    if station_passenger_arrive_30min[opt_num,j] >= 60 and station_passenger_arrive_30min[opt_num,j] < 80:
        sto_set_2[:,j] = stats.truncnorm((-60 - 2.22749) / 13.0313,10000000000, loc=2.22749, scale=13.0313).rvs(sample_num_N_2)
    if station_passenger_arrive_30min[opt_num,j] >= 80 and station_passenger_arrive_30min[opt_num,j] < 100:
        sto_set_2[:,j] = stats.truncnorm((-80 - 3.31231) / 16.2504,10000000000, loc=3.31231, scale=16.2504).rvs(sample_num_N_2)
    if station_passenger_arrive_30min[opt_num,j] >= 100 and station_passenger_arrive_30min[opt_num,j] < 400:            
        sto_set_2[:,j] = stats.truncnorm((-100- 6.87418) / 25.9947,10000000000, loc=6.87418, scale=25.9947).rvs(sample_num_N_2)

for k in range(sample_num_N_2):
    sto_set_2[k,263] = sum(sto_set_2[k,0:263]) # The sum of uncertainty samples across all regions
        
# Identify the optimal solution corresponding to the best optimal value within the M groups
x_best = np.where(result_obj_all[:] == min(result_obj_all[:]))[0][0]

# Substitute into the optimazation model with sample size N'
y_test_N = np.zeros((sample_num_N_2,263))
result_test_x_sum_out = np.zeros(263)
result_test_x_sum_in = np.zeros(263)
station_vehicle_redundant_sum = sum(station_vehicle_redundant[opt_num,:,9])

for n in range(263):
    result_test_x_sum_out[n] = sum(result_test_x_all[x_best, n, m] for m in i_iter)
    result_test_x_sum_in[n] =  sum(result_test_x_all[x_best, m, n] for m in i_iter)
    for sto_scene in range(sample_num_N_2):
    
        y_test_N[sto_scene,n] = abs((station_vehicle_redundant[opt_num, n, 9] - result_test_x_sum_out[n] + result_test_x_sum_in[n]) - station_passenger_redundant[opt_num,n,9] -
                                    gamma * max(1, 1*(station_vehicle_redundant_sum / (station_passenger_arrive_30min_allzone[opt_num]+ sto_set_2[sto_scene,263]))) * (station_passenger_arrive_30min[opt_num, n] + sto_set_2[sto_scene,n])) 

# Estimate the upper bound of the optimal objective value of the original problem
result_obj_upperbound = (alpha1* 1/sample_num_N_2*(sum(y_test_N[sto_scene,n] for n in range(263) for sto_scene in range(sample_num_N_2)))+ 
                         alpha1* 1/sample_num_N_2*(sum(y_test_N[sto_scene,0] for sto_scene in range(sample_num_N_2)))+ #Since Newark Airport is located at a considerable distance, 10 km can be added to the original dispatching distance of Newark Airport
                         alpha2* sum(taxi_zones_centroid_route_distance[m, n]/1000 * result_test_x_all[x_best, m, n] for m in range(263) for n in range(263)) + 
                         alpha3* sum(result_test_beta_all[x_best, n] for n in range(263)))

# Estimate the variance
result_obj_minus = np.zeros(sample_num_N_2)
result_test_beta_all_sum = sum(result_test_beta_all[x_best, n] for n in range(263))
result_test_distance_sum = sum(taxi_zones_centroid_route_distance[m, n]/1000 * result_test_x_all[x_best, m, n] for m in range(263) for n in range(263))
for sto_scene in range(sample_num_N_2):
    result_obj_minus[sto_scene] = (alpha1 * (sum(y_test_N[sto_scene,n] for n in range(263)))+ 
                                   alpha1 * (y_test_N[sto_scene,0])+ #Since Newark Airport is located at a considerable distance, 10 km can be added to the original dispatching distance of Newark Airport
                                   alpha2 *  result_test_distance_sum + 
                                   alpha3 * result_test_beta_all_sum) - result_obj_upperbound

result_obj_var_2 = 1/(sample_num_N_2* (sample_num_N_2 - 1))* sum(result_obj_minus[:]**2)

# The optimal solution can be used as the vehicle rebalancing decision for the current minute
dispatch_vehicle_everymin = np.zeros((263,263))
for j in range(263):
    for k in range(263):
        dispatch_vehicle_everymin[j,k] = result_test_x_all[x_best,j,k] 

# Record the upper bound, lower bound, and variance
lowerbound = result_obj_lowerbound
upperbound = result_obj_upperbound
var1 = result_obj_var_1
var2 = result_obj_var_2

# Calculate the gap between the lower and upper bounds, and the variance and standard deviation of the gap
gap_record = upperbound - lowerbound
gap_percent_record = (upperbound - lowerbound)/ (lowerbound + 0.0000000000001)
var_record = var1 + var2
std_record = math.sqrt(var_record)

print("gap_record = " , gap_record)
print("gap_percent_record = " , gap_percent_record)
print("var_record = " , var_record)
print("std_record = " , std_record)











