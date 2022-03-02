#!/usr/bin/env python
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros
from skimage.draw import circle

#debug
import matplotlib.pyplot as plt

#make my life easy
from scipy.spatial.distance import cdist

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils


TRANS_GOAL_TOL = .1  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .3  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.025, 0.13, 0.26]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 10#16#5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = 1#1.6#.025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'path.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
TEMP_HARDCODE_PATH = [[0,0,0],[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
#TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        map = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        self.map_resolution = round(map.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        self.map_nonzero_idxes = np.argwhere(self.map_np)

        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, 'path_complete.npy')).T
        #self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)


        # Ben: extra variables
        self.coord_error_tolerance = 1e-6#1e-8     # for coordinates
        self.vel_max = TRANS_VEL_OPTS[-1]
        self.rot_vel_max = ROT_VEL_OPTS[-1]
        self.desirable_trans_vel = 0.13

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()
    
    def trajectory_rollout(self, vel, rot_vel):           
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
        
        # Responsible: Ben
        # assume the robot is initially positioned along the x-axis at the origin. pose = [0,0,0]

        traj_points = np.zeros((3, self.horizon_timesteps))
        if abs(rot_vel - 0) < self.coord_error_tolerance:
            # drive on a straight line
            traj_points[0,:] = vel * np.arange(1, self.horizon_timesteps + 1)
        elif  abs(vel - 0) < self.coord_error_tolerance:
            traj_points[2,:] = rot_vel * np.arange(1, self.horizon_timesteps + 1)
        else:
            radius = abs(vel/rot_vel)
            substep_arc = abs(vel) * INTEGRATION_DT
            substep_angle = substep_arc/radius
            forward = vel > 0
            upward = (vel > 0) == (rot_vel > 0)         # 1st and 2nd quadrant
            heading_upward = rot_vel > 0

            # calcualate the exact position of the robot (not through linearization)
            for t in range(1, self.horizon_timesteps + 1):
                traj_points[0, t-1] = radius * np.math.sin(substep_angle * t)
                traj_points[1, t-1] = radius * (1 - np.math.cos(substep_angle * t))
                traj_points[2, t-1] = substep_angle * t

            if not forward:
                traj_points[0, :] = -1 * traj_points[0, :]
            if not upward:
                traj_points[1, :] = -1 * traj_points[1, :]
            if not heading_upward:
                traj_points[2, :] = -1 * traj_points[2, :]
        return traj_points 
    

    def robot_controller_exact(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")
        
        # Responsible: Ben
        # calculate radius of motion, or straight line along x-axis -> determine velocities exactly analytically, with constraints enforced on both velocities

        x_i = node_i[0,0]
        y_i = node_i[1,0]

        x_s = point_s[0,0]
        y_s = point_s[1,0]

        if abs(y_s - y_i) <  self.coord_error_tolerance:
            # drive in a straight line
            if abs(x_s - x_i) <  self.coord_error_tolerance: #JONATHAN: Play with this parameter adaptively as we are further away from the waypoint
                # start and end point coincides
                return 0, 0
            else:
                # try to get to the goal location after all substeps
                ideal_trans_vel = (x_s - x_i) / CONTROL_HORIZON
                return np.clip(ideal_trans_vel, -self.vel_max, self.vel_max), 0
        elif abs(x_s - x_i) <  self.coord_error_tolerance:
            # goal is perpendicular to the current viable robot path -> use circular path
            # this is a special case due to the perpenducularity
            radius = abs(y_s - y_i)/2  
            arc_angle = np.math.pi
        else:
            # drive along the arc of a circle with radius > 0
            y_c = ((x_s - x_i)**2 / (y_s - y_i) + y_i + y_s) * 1/2
            radius = abs(y_c - y_i)     # radius = translational / rotational vel
            arc_angle = np.math.asin(x_s/radius)
            
        arc_length = arc_angle * radius     # may be negative if the robot has to move backward


        ideal_trans_vel = self.desirable_trans_vel                  # want to move forward at this speed
        viable_trans_vel =  np.clip(ideal_trans_vel, -self.vel_max, self.vel_max)

        trans_vel_mag = abs(viable_trans_vel)
        forward = viable_trans_vel > 0
        rot_vel_mag = trans_vel_mag/radius
        if rot_vel_mag > self.rot_vel_max:
            rot_vel_mag =  np.clip(rot_vel_mag, 0, self.rot_vel_max)
            trans_vel_mag = rot_vel_mag * radius

        if forward:
            if y_s > y_i:
                return trans_vel_mag, rot_vel_mag   #1st quadrant
            else:
                return trans_vel_mag, -rot_vel_mag  #4th quadrant
        else:
            if y_s > y_i:
                return -trans_vel_mag, -rot_vel_mag #2nd quadrant
            else:
                return -trans_vel_mag, rot_vel_mag  #3rd quadrant


    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            # start trajectory rollout algorithm
            # print("TO DO: Propogate the trajectory forward, storing the resulting points in local_paths!")
            
            # prepare all coordinate transformations
            #print(self.pose_in_map_np)
            theta_v_w = self.pose_in_map_np[2]
            T_w_v = np.array([[np.math.cos(theta_v_w), -np.math.sin(theta_v_w), self.pose_in_map_np[0]],
                            [np.math.sin(theta_v_w),  np.math.cos(theta_v_w), self.pose_in_map_np[1]],
                            [0, 0, 1]])    #ece470 convention
            T_v_w = np.linalg.inv(T_w_v)
            
            #find collision free controls and cost

            # #HYPERPARAMTERS
            discount_factor = 0.9 #Larger means more emphasis on next waypoint rather than later waypoints
            obstacle_avoid_factor = 1.03 #larger penalizes being near obstacles more
            increaseBubble = 2 #the multiplier on the minimal radius for extra collision checking
            rot_error_factor = 3

            # find goal trajectory:
            # current goal in current robot frame

            cur_goal_vector = np.resize(self.cur_goal, (-1, 1))
            cur_goal_x = self.cur_goal[0]
            cur_goal_y = self.cur_goal[1]
            cur_goal_v = np.matmul(T_v_w, np.array([[cur_goal_x], [cur_goal_y], [1]]))
            trans_vel_goal, rot_vel_goal = self.robot_controller_exact(np.zeros((3,1)), cur_goal_v)

            goal_traj_v = self.trajectory_rollout(trans_vel_goal, rot_vel_goal)        # robot trajectory as seen in frame of node_i
            goal_traj = np.matmul(T_w_v, np.vstack((goal_traj_v[:2, :], np.ones((1, self.horizon_timesteps)))))      # convert x,y coord from frame i to world frame
            goal_traj[2, :] = (theta_v_w + goal_traj_v[2, :]) % (2 * np.math.pi)
                

            dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
            abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
            rot_error = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)

            collision_free_controls = []
            collision_free_costs = []
            collision_free_trajs = []

            near_obstacle_penalty = []

            for trans_vel, rot_vel in self.all_opts:
                robot_traj_v = self.trajectory_rollout(trans_vel, rot_vel)        # robot trajectory as seen in frame of node_i
                robot_traj = np.matmul(T_w_v, np.vstack((robot_traj_v[:2, :], np.ones((1, self.horizon_timesteps)))))      # convert x,y coord from frame i to world frame
                #note: at this point, havent calculated the exact heading yet

                local_paths_pixels = (self.map_origin[:2] + robot_traj[:2, :].T) / self.map_resolution
                
                hasCollision = False
                foundExtraCollision = False
               
                numOfCollisions = 0

                for timestep in range(local_paths_pixels.shape[0]):
                    xp = int(local_paths_pixels[timestep,0])
                    yp = int(local_paths_pixels[timestep,1])
                    #print("xp: ",xp)
                    #print("yp: ",yp)

                    #radius to check
                    r = self.collision_radius_pix
                    r = int(r)
                    

                    #generate pixel points within circle
                    xcirc, ycirc = circle(xp,yp,r)
                    circlepoints = np.array([xcirc,ycirc]).T
                    #numOfCirclePoints = circlepoints.shape(0) #number of rows

                    xcirc, ycirc = circle(xp,yp,increaseBubble + r)
                    circlepoints_extra = np.array([xcirc,ycirc]).T


                    if any(self.map_np[circlepoints[:,1], circlepoints[:,0]] == 100):
                        hasCollision = True
                        break

                    if any(self.map_np[circlepoints_extra[:,1], circlepoints_extra[:,0]] == 100): #and not foundExtraCollision:
                        foundExtraCollision = True
                        numOfCollisions += np.count_nonzero(self.map_np[circlepoints_extra[:,1], circlepoints_extra[:,0]] == 100)*(local_paths_pixels.shape[0]/(10*timestep+1))
                                                  
                if not hasCollision:
                    # this is a good trajectory -> calculate heading & save
                    robot_traj[2, :] = (theta_v_w + robot_traj_v[2, :]) % (2 * np.math.pi)
                    # disp_to_goal_node = robot_traj - self.cur_goal
                    disp_to_goal_node = robot_traj[:2] - goal_traj[:2]
                    dist_to_goal_node = np.abs(disp_to_goal_node).sum(axis = 0)
                    
                    discounted_weight = np.power(discount_factor, np.arange(1, self.horizon_timesteps + 1))

                    discounted_dist_to_goal_node = discounted_weight * dist_to_goal_node
                    dist_to_goal_node_total = discounted_dist_to_goal_node.sum()           # sum of all distances to cur_goal of the nodes in each trajectory
                    
					#My addition
                    if foundExtraCollision:
                        print("factor: ", obstacle_avoid_factor*(1 + numOfCollisions/100)) #I tried log10 (number of collisions/10)
                        dist_to_goal_node_total += obstacle_avoid_factor*(1 + numOfCollisions/100) + rot_error*rot_error_factor
                    
                    collision_free_controls.append([trans_vel, rot_vel])
                    collision_free_costs.append(dist_to_goal_node_total)
                    collision_free_trajs.append(robot_traj)

            best_traj_idx = np.argmin(collision_free_costs)
            control = collision_free_controls[best_traj_idx]
            # control = np.array([0,0])
            # print(collision_free_trajs[best_traj_idx])
            # self.local_path_pub.publish(utils.se2_pose_list_to_path(collision_free_trajs[best_traj_idx].T, 'map'))
            self.local_path_pub.publish(utils.se2_pose_list_to_path(goal_traj.T, 'map'))

            
            
            
            
            
            # for t in range(1, self.horizon_timesteps + 1):
			# 	# propogate trajectory forward, assuming perfect control of velocity and no dynamic effects
			# 	# propogate trajectory forward, assuming perfect control of velocity and no dynamic effects. #We have pose (self.pose_in_map_np) and we have vels from rows in self.all_opts
            #     #[[translation vel, roational vel]]
            #     #Local paths is a #of timesteps by #of vel rows by 3 (for points) matrix
            #     #horizon time steps are the number of timesteps we are propagating for. Therefore we come up with that amount of points along the trajectory
            #     #pose is stored in self.pose_in_map_np as [x,y,theta]

            #     for u in range(0,self.num_opts): #for each row of the all_opts matrix
            #         #Get point in last time frame
            #         x0 = local_paths[t-1,u,0]
            #         y0 = local_paths[t-1,u,1]
            #         the0 = local_paths[t-1,u,2]


            #         #Get velocities
            #         v = self.all_opts[u,0]
            #         w = self.all_opts[u,1]

            #         dt = INTEGRATION_DT

            #         # if u == 0 and t == 1:
            #         #     print(x0)
            #         #     print(y0)
            #         #     print(the0)

            #         if abs(w - 0) < 1e-8: #If no rotationa component
            #             # drive on a straight line
            #             xt = v * dt * np.math.cos(the0)
            #             yt = v * dt * np.math.sin(the0)
            #             thet = 0

            #         elif abs(v - 0) < 1e-8: #If no translational component
            #             #rotate on the spot
            #             xt = 0
            #             yt = 0
            #             thet = w*dt

            #         else:
            #             #Calculate values used multiple times
            #             radius = abs(v/w)
            #             substep_arc = abs(v) * dt
            #             substep_angle = substep_arc/radius

            #             #These deal with the whole CAST quadrant thing
            #             forward = v > 0
            #             upward = (v > 0) == (w > 0)
            #             heading_upward = w > 0

            #             #Displacement in world coordinates (inertial reference frame) after time step dt
            #             xt = radius * np.math.sin(substep_angle * t)
            #             yt = radius * (1 - np.math.cos(substep_angle * t))
            #             thet = substep_angle * t

            #             #Dealing with the whole CAST thing regarding sin and cos
            #             if not forward:
            #                 xt = -1 * xt
            #             if not upward:
            #                 yt = -1 * yt
            #             if not heading_upward:
            #                 thet = -1 * thet
                    
            #         #debugging
            #         # if(np.isnan(xt) and not caught):
            #         #     caught = True
            #         #     print("u: ",u)
            #         #     print("t: ",t)
            #         # elif(np.isnan(yt) and not caught):
            #         #     caught = True
            #         #     print("u: ",u)
            #         #     print("t: ",t)
            #         # elif(np.isnan(thet) and not caught):
            #         #     caught = True
            #         #     print("u: ",u)
            #         #     print("t: ",t)

            #         #Add displacement in world coordinates with initial conditions (last time step)
            #         xt = x0 + xt
            #         yt = y0 + yt
            #         thet = the0 + thet

            #         #Set local path
            #         local_paths[t,u,0] = xt
            #         local_paths[t,u,1] = yt
            #         local_paths[t,u,2] = thet
                    

            
            # #print(local_paths)
            # # check all trajectory points for collisions
            # # first find the closest collision point in the map to each local path point
            


            # local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution





            # #print(local_paths_pixels)
            # valid_opts = range(self.num_opts)
            # local_paths_lowest_collision_dist = np.ones(self.num_opts) * 50
            # #print(valid_opts)
            # #print(np.shape(self.map_np))
            # #print(local_paths_pixels)
            # #print(self.map_nonzero_idxes)
            # #print("TO DO: Check the points in local_path_pixels for collisions")
            # #print(range(local_paths_pixels.shape[1]))
            # for opt in range(local_paths_pixels.shape[1]):
            #     #print(opt)
            #     for timestep in range(local_paths_pixels.shape[0]):
            #         found = False
            #         #print("option: ", opt)
            #         #print("timestep: ", timestep)
            #         #We have x y pixel values in local_paths_pixels
            #         #Valid_opts is a vector counting to the number of available trajectories
            #         #We want to save the lowest collision distance out of all of the trajectory points in local_paths_lowest_collision_dist
            #         #Where are the map pixels stored? 0s are obstacles in self.map_np 1600 by 1600 matrix. 
            #         #self.collision_radius_pix is the minimum radius to avoid collision
            #         #self.map_nonzero_idxes contains a matrix whose rows [x y] correspond to the pixels where there is an obstacle.

            #         #First get pixel point
					



            #         xp = int(local_paths_pixels[timestep,opt,0])
            #         yp = int(local_paths_pixels[timestep,opt,1])
            #         #print("xp: ",xp)
            #         #print("yp: ",yp)

            #         #radius to check
            #         r = self.collision_radius_pix
            #         r = int(r)
                    

            #         #generate pixel points within circle
            #         xcirc, ycirc = circle(xp,yp,r)
            #         circlepoints = np.array([xcirc,ycirc]).T
            #         #numOfCirclePoints = circlepoints.shape(0) #number of rows

            #         obstaclelist = self.map_nonzero_idxes
            #         if any(self.map_np[circlepoints[:,1], circlepoints[:,0]] == 100):
            #             found = False
            #             #print("break")
            #             valid_opts.remove(opt)
            #             #print(valid_opts)
            #             break
            
            # # # remove trajectories that were deemed to have collisions
            # # #print("TO DO: Remove trajectories with collisions!")

            # # calculate final cost and choose best option
            # #print("TO DO: Calculate the final cost and choose the best control option!")
    

            # # #HYPERPARAMTERS
            # discount_factor = 0.9 #Larger means more emphasis on next waypoint rather than later waypoints


            # if len(valid_opts) == 0:
            #     pass
            # else:
            #     # disp_to_goal_node = local_paths[:,valid_opts,:2] - self.cur_goal[:2]
            #     disp_to_goal_node = local_paths[:,valid_opts] - self.cur_goal
            #     dist_to_goal_node = np.abs(disp_to_goal_node).sum(axis = 2)
                
            #     discounted_weight = np.power(discount_factor, np.resize(np.arange(1, self.horizon_timesteps + 3), (-1, 1)))

            #     discounted_dist_to_goal_node = discounted_weight * dist_to_goal_node
            #     dist_to_goal_node_total = discounted_dist_to_goal_node.sum(axis = 0)           # sum of all distances to cur_goal of the nodes in each trajectory
                
            #     best_opt = valid_opts[np.argmin(dist_to_goal_node_total)]
            #     control = self.all_opts[best_opt,:]

                # print('hi')
                
                # local_paths[:,valid_opts,:2]

            


            # final_cost = np.zeros(self.num_opts)
            
            # # #HYPERPARAMTERS
            # hypParam1 = 0.9 #Larger means more emphasis on next waypoint rather than later waypoints

            # if final_cost.size == 0:
            #     pass
            # else:
            #     #Get list of waypoints
            #     #print(self.path_tuples) #stored as rows [x y theta]
            #     waypoints = self.path_tuples[:,:2]
            #     #print("waypoints: ", waypoints)
            #     #Get list of nodes and their L2 distances to the waypoints
            #     #print(valid_opts)
            #     for j in valid_opts: #cycle through the valid options
            #         traj = local_paths[:,j,:2]
                    
            #         distances = cdist(traj,waypoints)
            #         #print(distances)
            #         #In distances, each row is for each node, each column is the distance to each waypoint. Ex. Row 0 contains the distances of node 0 to each waypoint
            #         #We only want to count the waypoints after the one we are closest to. So find argmin of the first node row
            #         closest_waypoint2node0 = np.argmin(distances[0,:])
					
            #         #only look at distances past the closest one to node 0
            #         valid_WP_D = distances[:,closest_waypoint2node0:]
            #         #print(valid_WP_D)
            #         #now we have a matrix of valid distances from the next waypoint onward with each of our nodes --> validwaypointdistances (nodes by valid waypoints)    
                    
            #         #Find out the index of the closest waypoint to each node
            #         #valid_WP_mapping = np.argmin(valid_WP_D,1)

            #         #I ues this multiple times, so to save time, I do it once here.
            #         numCols_valid_WP = np.size(valid_WP_D,1)

            #         #weighted sum the rows of the matrix, first we need to define a weighting column vector the size of the number of waypoints remaining
            #         weights = np.arange(1,numCols_valid_WP+1).reshape(numCols_valid_WP,1)
            #         weights = 1/np.exp(-hypParam1*weights)

            #         #Apply weights as per the mapping to each nodes nearest waypoint. The further the waypoint, the less value that part of the score should have.
            #         #weights = weights[valid_WP_mapping]
            #         #print(weights)
            #         #print(valid_WP_D.T)
            #         #final_cost[j] = np.matmul(valid_WP_D.T,weights)[0]
                   
            #         final_cost[j] = np.sum(np.matmul(valid_WP_D,weights))
            #         #Weighted sum the shortest distances, penalizing the further waypoints

            
            
            # if final_cost.size == 0:  # hardcoded recovery if all options have collision
            #     control = [-.1, 0]
            # else:
            #     #print(np.size(self.all_opts))
            #     #print(np.size(valid_opts))
            #     #print(final_cost.argmax())
            #     #print(final_cost)
            #     best_opt = final_cost.argmin()
               
            #     #print(local_paths[:5, best_opt,:])
            #     control = self.all_opts[best_opt,:]
            #     # control = self.all_opts[-1,:]
                
            # #     plt.imshow(self.map_np)


            # # #for plots in range(0,self.num_opts):
            # # #    path1x = local_paths[:,plots,0]
            # # #    path1y = local_paths[:,plots,1]
            
            # # #    plt.scatter(path1y,path1x)
            # #     plt.scatter(utils.se2_pose_list_to_path(local_paths[:, best_opt], 'map'))
            # #     plt.show()


            # control = self.all_opts[best_opt,:]
            # # control = np.array([0,0])

            # for i in range(local_paths.shape[1]):
            #     self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, i], 'map'))

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
                 control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass
