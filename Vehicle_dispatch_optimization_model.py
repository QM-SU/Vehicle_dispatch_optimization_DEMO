
def dispatch_optimization(sample_num_N, sample_iter, opt_num, station_vehicle_redundant, station_passenger_redundant, station_passenger_arrive_30min_allzone, station_passenger_arrive_30min,  sto_set, taxi_zones_neighbor_10min_extension, taxi_zones_centroid_route_distance):
    import numpy as np
    import gurobipy
    
    # Optimization parameters
    alpha1 = 10
    alpha2 = 1
    alpha3 = 100
    
    gamma = 1/2
    P_wait_max = 0
        
    # Create the vehicle dispatch model
    dispatch_test = gurobipy.Model('dispatch_test')
    
    # Create variables
    i_iter = list(range(263))
    j_iter = list(range(263))
    k_iter = list(range(10))
    sto_scene_iter = list(range(sample_num_N))
    
    
    x_test = dispatch_test.addVars(i_iter, j_iter, vtype=gurobipy.gurobipy.GRB.INTEGER)
    beta_test = dispatch_test.addVars(i_iter, vtype=gurobipy.GRB.INTEGER)
    minus_test = dispatch_test.addVars(sto_scene_iter, i_iter, lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS)
    y_test = dispatch_test.addVars(sto_scene_iter, i_iter, lb=0, ub=gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS)
    
   
    
    # Update variable environment
    dispatch_test.update()
    
    
    # Create objective function    
    dispatch_test.setObjective(alpha1 * 1/sample_num_N * (sum(y_test[sto_scene,n] for n in j_iter for sto_scene in sto_scene_iter)) + 
                               alpha1 * 1/sample_num_N * (sum(y_test[sto_scene,0] for sto_scene in sto_scene_iter)) + #Since Newark Airport is located at a considerable distance, 10 km can be added to the original dispatching distance of Newark Airport
                               alpha2 * sum(taxi_zones_centroid_route_distance[m, n]/1000 * x_test[m, n] for m in i_iter for n in j_iter) + 
                               alpha3 * sum(beta_test[n] for n in j_iter),
                               gurobipy.GRB.MINIMIZE)


    # The objective of supply-demand gap in the objective function
    dispatch_test.addConstrs(minus_test[sto_scene,n] == ((station_vehicle_redundant[opt_num, n, 9] - sum(x_test[n, m] for m in i_iter) + sum(x_test[m, n] for m in i_iter)) - station_passenger_redundant[opt_num,n,9] -
                                                         gamma * max(1, 1*(sum(station_vehicle_redundant[opt_num,:,9]) / (station_passenger_arrive_30min_allzone[opt_num]+ sto_set[sample_iter,sto_scene,263]))) * (station_passenger_arrive_30min[opt_num, n] + sto_set[sample_iter,sto_scene,n])) for n in j_iter for sto_scene in sto_scene_iter)
    dispatch_test.addConstrs(y_test[sto_scene,n] == gurobipy.abs_(minus_test[sto_scene,n]) for n in j_iter for sto_scene in sto_scene_iter)
 
           
    # Create constraints    
    # The number of unfulfilled demands in each region after vehicle rebalancing should not be greater than the upper limit of waiting passengers (In the simulation, the upper limit of waiting passengers = 0)
    dispatch_test.addConstrs(station_passenger_redundant[opt_num,n,9] - station_vehicle_redundant[opt_num,n,9] + sum(x_test[n, m] for m in i_iter) - sum(x_test[m, n] for m in i_iter) <= P_wait_max + beta_test[n] for n in j_iter)
    
    # The vehicle can only be rebalanced to these regions for rebalancing to ensure timely arrival
    dispatch_test.addConstrs(x_test[m,n]* (1- taxi_zones_neighbor_10min_extension[opt_num,m,n]) <= 0 for m in i_iter for n in j_iter)
    
    # The vehicle supply region for rebalancing must ensure that there are surplus vehicles throughout the entire rebalancing time window to ensure that the service quality of the region itself is not affected
    dispatch_test.addConstrs(sum(x_test[m,n] for n in j_iter) <= station_vehicle_redundant[opt_num,m,k] for m in i_iter for k in k_iter)
    
    # Beta is greater than or equal to 0
    dispatch_test.addConstrs(beta_test[n] >= 0 for n in j_iter)
    
    # X is greater than or equal to 0
    dispatch_test.addConstrs(x_test[m,n] >= 0 for m in i_iter for n in j_iter)
    
    # Model optimization
    dispatch_test.optimize()
    
    # Output result
    result_test_x = np.zeros((263,263))
    result_test_beta = np.zeros(263)
    solution_x = dispatch_test.getAttr('x',x_test)
    solution_beta = dispatch_test.getAttr('x',beta_test)
    result_test_y = np.zeros((sample_num_N,263))
    solution_y = dispatch_test.getAttr('x',y_test)
    
    # Parameter matrix
    for p,q in solution_x.items():
        result_test_x[p[0],p[1]] = q
    for p,q in solution_beta.items():
        result_test_beta[p] = q
    for p,q in solution_y.items():
        result_test_y[p[0],p[1]] = q

    
    return(result_test_x, result_test_beta, result_test_y)

