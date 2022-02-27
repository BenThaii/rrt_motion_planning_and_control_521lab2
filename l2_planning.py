#!/usr/bin/env python
#Standard Libraries
import numpy as np
from scipy import argmin
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import circle
from scipy.linalg import block_diag

#debug
import matplotlib.pyplot as plt

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        try:
            self.occupancy_map = load_map(map_filename)[:,:,0]     #1: free; 0: occupied
        except:
            self.occupancy_map = load_map(map_filename)

        # self.occupancy_map = np.ones((1600,1600))
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[0] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[1] * self.map_settings_dict["resolution"]

        
        #Robot information
        self.robot_radius = 0.22 #m
        # self.vel_max = 0.5 #m/s (Feel free to change!)
        # self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        # Ben: this reflects the real robot more closely
        self.vel_max = 0.26 #m/s (Feel free to change!)
        self.rot_vel_max = 1.82 #rad/s (Feel free to change!)


        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.6 #s      #ben: max translation between consecutive pose is approx 0.4m 
        self.num_substeps = 10

        #Planning storage
        # self.nodes = [Node(np.zeros((3,1)), -1, 0)]
        self.nodes = [Node(np.array([[-0.5], [-18.5], [0]]), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        
        # Ben: vertorize and discretize environment information
        self.world_shape_vect =  np.array([[self.map_shape[0] * self.map_settings_dict["resolution"]], 
                                    [self.map_shape[1] * self.map_settings_dict["resolution"]]])
        self.map_shape_vect = np.array([[self.map_shape[0]], 
                                        [self.map_shape[1]]])

        self.lower_corner = np.array([[self.map_settings_dict["origin"][0]], 
                                      [self.map_settings_dict["origin"][1]]])
        self.robot_pixel_radius = np.ceil(self.robot_radius/ self.map_settings_dict["resolution"])  # upper bound for safety
        
        # do other repeated calculations
        self.traj_time = self.timestep*self.num_substeps
        self.max_dist_per_step = self.timestep*self.vel_max
        self.max_dist_per_traj = self.traj_time*self.vel_max
        self.max_dist_per_traj_sq = self.max_dist_per_traj**2
        self.stopping_dist_sq = self.stopping_dist ** 2

        # Ben: other configurations
        self.coord_error_tolerance = 1e-8     # for coordinates
        self.max_rrt_iteration = 5000 
        self.goal_sample_rate = 0.05  
        final_goal_sample_rate = 0.1
        self.goal_sample_rate_inc = (final_goal_sample_rate - self.goal_sample_rate)/self.max_rrt_iteration
        self.goal_region_sample_rate = 0.3
        self.box_ratio = 1
        box_shrink_start_ratio = 0.5
        self.box_shrink_start = self.max_rrt_iteration * box_shrink_start_ratio
        self.final_box_ratio = 0.15
        self.goal_region_box_ratio = 0.4
        self.shrink_ratio = 0.99
        self.num_closest_node_consideration = 10



        self.num_rollouts_per_node = 5
        exact_soln_radius =  self.vel_max * self.traj_time
        self.exact_soln_radius_sq = exact_soln_radius ** 2

        self.visitted_locations = np.zeros(self.map_shape)
        self.invalid_locations = np.zeros(self.map_shape)

        self.sampled_pts = []
        self.closest_pts = np.zeros((3, self.max_rrt_iteration))
        self.node_dist_scaling = [1]    # start with no distance scaling for the first node
        self.invalid_radius = self.robot_radius ** 2
        self.invalid_pts = 0
        self.invalid_pixel_radius = 1

        # debug:
        self.fig, self.axs = plt.subplots(1,2)
        self.axs[0].imshow(self.occupancy_map)
        return

    #Functions required for RRT
    def sample_map_space(self):
        # sample points in the map (constrained by a box that is shrinking by self.box_ratio)
        # Responsible: Ben
        # output: [x,y] coordinate to drive the robot towards

        # print("TO DO: Sample point to drive towards")
        left_buffer = (self.goal_point[0, 0] - self.bounds[0, 0]) * self.box_ratio
        right_buffer = (self.bounds[0, 1] - self.goal_point[0, 0]) * self.box_ratio
        bottom_buffer = (self.goal_point[1, 0] - self.bounds[1, 0]) * self.box_ratio
        top_buffer = (self.bounds[1, 1] - self.goal_point[1, 0]) * self.box_ratio

        
        rdn_point = np.zeros((2, 1))
        rdn_point[0,0] = self.goal_point[0, 0] - left_buffer + np.random.random() * (left_buffer+right_buffer)
        rdn_point[1,0] = self.goal_point[1, 0] - bottom_buffer + np.random.random() * (bottom_buffer+top_buffer)

        return rdn_point

    def sample_goal_region_space(self):
        # sample points close to the goal
        # Responsible: Ben
        # output: [x,y] coordinate to drive the robot towards
        left_buffer = (self.goal_point[0, 0] - self.bounds[0, 0]) * self.goal_region_box_ratio
        right_buffer = (self.bounds[0, 1] - self.goal_point[0, 0]) * self.goal_region_box_ratio
        bottom_buffer = (self.goal_point[1, 0] - self.bounds[1, 0]) * self.goal_region_box_ratio
        top_buffer = (self.bounds[1, 1] - self.goal_point[1, 0]) * self.goal_region_box_ratio

        
        rdn_point = np.zeros((2, 1))
        rdn_point[0,0] = self.goal_point[0, 0] - left_buffer + np.random.random() * (left_buffer+right_buffer)
        rdn_point[1,0] = self.goal_point[1, 0] - bottom_buffer + np.random.random() * (bottom_buffer+top_buffer)

        return rdn_point
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        # print("TO DO: Check that nodes are not duplicates")

        #Responsible: Ben
        # this code can be improved further with a KD tree; Duplicated -> return True
        for node in self.nodes:
            if abs(point[0,0] - node.point[0,0]) < self.coord_error_tolerance:
                if abs(point[1,0] - node.point[1,0]) < self.coord_error_tolerance:   # do this to avoid redundant checks
                    return True
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        # print("TO DO: Implement a method to get the closest node to a sampled point")
        
        #Responsible: Ben
        # return closest nodes that are not invalid

        # obtain distance to all nodes in the tree
        d_sq_list = [((self.nodes[i].point[0,0] - point[0,0])**2 + (self.nodes[i].point[1,0] - point[1,0])**2) * self.node_dist_scaling[i]
                 for i in range(len(self.nodes))]
        
        # refine the selected points when close to (global) goal point:
        if np.abs(point - self.goal_point).sum() < 2 * self.coord_error_tolerance:
            d_sq_list_np = np.array(d_sq_list)
            idx = np.where(d_sq_list_np < self.max_dist_per_traj_sq)[0]
            if len(idx) == 0:
                # no nodes within a desired distance from the desired point -> return the closest node
                return d_sq_list.index(min(d_sq_list))
            else:
                # for nodes within the desired distance from the desired point, select node whose straight line trajectory is not intercepted by obstacles -> improve the quality of points 
                k = min(len(idx), self.num_closest_node_consideration)      # dont want to consider too many points to increase speed and promote sparsity -> explore more ways to get to the goal
                idx = np.argpartition(d_sq_list, k - 1)
                for node_idx in idx:
                    point_i = self.nodes[node_idx].point[:2,[0]]

                    straight_traj = self.straight_line_traj(point_i, point)
                    straight_traj_cells_x, straight_traj_cells_y = self.point_to_cell(straight_traj)
                    if any(self.occupancy_map[straight_traj_cells_y, straight_traj_cells_x] == 0):
                        # obstacle intercepts straight line distance
                        continue
                    else:
                        # print('hi')
                        self.axs[0].plot(straight_traj_cells_x, straight_traj_cells_y)
                        return node_idx
                # if no collision-free path is found, return a random node
                # print('sampled randomly')
                return np.random.choice(np.where(d_sq_list_np != np.inf)[0])

        else:
            return d_sq_list.index(min(d_sq_list))

        # backup: in case no valid point whose straight line distance is found, return the closest point
        return d_sq_list.index(min(d_sq_list))

    def straight_line_traj(self, point_i, point_s):
        # Responsible: Ben
        # helper function that returns the straight line trajectory from point_i to point_s
        # output ->traj: 3xnum_substeps matrix containing milestones of the trajectory
        disp_vect = point_s - point_i
        distance = np.linalg.norm(disp_vect)
        dir_vect = disp_vect/distance
        num_mile_stones = np.math.floor(distance/self.map_settings_dict["resolution"])
        traj_points = np.tile(np.arange(1,num_mile_stones), (3,1))
        traj_points = point_i + traj_points[:2,:] * dir_vect * self.map_settings_dict["resolution"]
        return traj_points


    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does have many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # print("TO DO: Implment a method to simulate a trajectory given a sampled point")

        # Responsible: Ben
        # output: 
        #       traj_foound: True of a good trajectory is found
        #       traj: 3xnum_substeps matrix containing milestones of the trajectory
        #       traj_velocity: the velocity of the robot while traversing the trajectory -> for cost calculation

        # find trajectory in node_i's frame, then convert back to world frame
        theta_i_w = node_i[2, 0]
        T_w_i = np.array([[np.math.cos(theta_i_w), -np.math.sin(theta_i_w), node_i[0,0]],
                          [np.math.sin(theta_i_w),  np.math.cos(theta_i_w), node_i[1,0]],
                          [0, 0, 1]])    #ece470 convention
        T_i_w = np.linalg.inv(T_w_i)

        node_i_i = np.zeros((3, 1))
        point_s_i_aug = np.matmul(T_i_w,  np.vstack((point_s, 1)))
        point_s_i = point_s_i_aug[:2]

        x_s = point_s[0,0]
        y_s = point_s[1,0]


        # check simple straight-line scenerios
        if abs(y_s) <  self.coord_error_tolerance * x_s:        # allow more error for far-away points
            # drive in a straight line
            if abs(x_s) <  self.coord_error_tolerance:
                # start and end point coincides -> has already reached its goal
                vel = 0
                rot_vel = 0
            else:
                # try to get to the goal location after all substeps
                ideal_trans_vel = (x_s) / self.traj_time
                vel = np.clip(ideal_trans_vel, -self.vel_max, self.vel_max)
                rot_vel = 0
            
            robot_traj_i = self.trajectory_rollout(vel, rot_vel)        # robot trajectory as seen in frame of node_i
            robot_traj = np.matmul(T_w_i, np.vstack((robot_traj_i[:2, :], np.ones((1, self.num_substeps)))))      # convert x,y coord from frame i to world frame
            if not self.traj_has_collision(robot_traj):
                # do not have any collision
                robot_traj[2, :] = (theta_i_w + robot_traj_i[2, :]) % np.math.pi
                return True, robot_traj, vel
            

        elif x_s ** 2 + y_s ** 2 < self.exact_soln_radius_sq:
            # close to the goal, try analytical solution first
            vel, rot_vel = self.robot_controller_exact(node_i_i, point_s_i)
            robot_traj_i = self.trajectory_rollout(vel, rot_vel)        # robot trajectory as seen in frame of node_i
            robot_traj = np.matmul(T_w_i, np.vstack((robot_traj_i[:2, :], np.ones((1, self.num_substeps)))))      # convert x,y coord from frame i to world frame
            if not self.traj_has_collision(robot_traj):
                # do not have any collision
                robot_traj[2, :] = (theta_i_w + robot_traj_i[2, :]) % np.math.pi
                return True, robot_traj, vel

        # neither easy, nor exact solution works -> use random rollout to select a trajectory
        best_traj = np.zeros((3, self.num_substeps))
        best_dist_from_goal_culmulative = np.inf
        best_vel = 0
        for _ in range(self.num_rollouts_per_node):
            vel, rot_vel = self.robot_controller(node_i_i, point_s_i)   #Ben: calculate velocities using coordinates in point_s frame
            robot_traj_i = self.trajectory_rollout(vel, rot_vel)        # robot trajectory as seen in frame of node_i
            robot_traj = np.matmul(T_w_i, np.vstack((robot_traj_i[:2, :], np.ones((1, self.num_substeps)))))      # convert x,y coord from frame i to world frame
            #note: at this point, havent calculated the exact heading yet
            
            # discard when final point of the trajectory has already been visitted, or out of boundary
            terminal_node_cell = self.point_to_cell(robot_traj[:2,[-1]])
            if terminal_node_cell[0,0] < 0 or terminal_node_cell[0,0] >= self.map_shape[0] or terminal_node_cell[1,0] < 0 or terminal_node_cell[1,0] >= self.map_shape[1]: 
                # final point of trajectory is outside of the boundary
                continue
            if self.visitted_locations[terminal_node_cell[1,0], terminal_node_cell[0,0]] == 1:
                # dont consider trajectories that end in visitted states
                continue
            
            # discard trajectories that end at to invalid states
            new_idx_x, new_idx_y = circle(terminal_node_cell[0,0], terminal_node_cell[1,0], self.invalid_pixel_radius) 
            if any(self.invalid_locations[new_idx_y, new_idx_x] == 1):
                continue

            # discard when trajectory has a collision
            if self.traj_has_collision(robot_traj):
                continue

            
            # keep the best trajectory
            deviations_from_goal = (robot_traj[:2, :] - point_s)
            dist_from_goal_culmulative = (deviations_from_goal * deviations_from_goal).sum()
            if dist_from_goal_culmulative < best_dist_from_goal_culmulative:
                best_dist_from_goal_culmulative = dist_from_goal_culmulative
                robot_traj[2, :] = (theta_i_w + robot_traj_i[2, :]) % np.math.pi
                best_traj = robot_traj
                best_vel = vel

        if best_dist_from_goal_culmulative == np.inf:
            # has not found any path without collision
            return False, None, None

        return True, best_traj, best_vel

    def traj_has_collision(self, robot_traj):
        # use the x and y coordinates of the trajectory, return True if there is a collision
        # responsible: Ben (helper function)
        footprint_x, footprint_y = self.points_to_robot_circle(robot_traj[:2,:])

        footprint_x_np = np.array(footprint_x)
        footprint_y_np = np.array(footprint_y)

        if any(footprint_x_np < 0) or any(footprint_x_np >= self.map_shape[0]) or any(footprint_y_np < 0) or any(footprint_y_np >= self.map_shape[1]): 
            # there is a collision against the boundary
            return True

        elif any(self.occupancy_map[footprint_y, footprint_x] == 0):
            # there is a collision agains the obstacles
            return True
        else:
            return False

    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")
        
        # Responsible: Ben
        # generate random velocity control within the velocity constraints
        vel = -self.vel_max + np.random.random() * 2 * self.vel_max
        rot_vel = -self.rot_vel_max + np.random.random() * 2 * self.rot_vel_max
        return vel, rot_vel


        


    
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
            if abs(x_s - x_i) <  self.coord_error_tolerance:
                # start and end point coincides
                return 0, 0
            else:
                # try to get to the goal location after all substeps
                ideal_trans_vel = (x_s - x_i) / self.traj_time
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
        ideal_trans_vel = arc_length/ self.traj_time
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


    
    def trajectory_rollout(self, vel, rot_vel):           
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
        
        # Responsible: Ben
        # assume the robot is initially positioned along the x-axis at the origin. pose = [0,0,0]

        traj_points = np.zeros((3, self.num_substeps))
        if abs(rot_vel - 0) < self.coord_error_tolerance:
            # drive on a straight line
            traj_points[0,:] = vel * range(1, self.num_substeps + 1)
        else:
            radius = abs(vel/rot_vel)
            substep_arc = abs(vel) * self.timestep
            substep_angle = substep_arc/radius
            forward = vel > 0
            upward = (vel > 0) == (rot_vel > 0)         # 1st and 2nd quadrant
            heading_upward = rot_vel > 0

            # calcualate the exact position of the robot (not through linearization)
            for t in range(1, self.num_substeps + 1):
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
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        
        # Responsible: Ben
        # Output: (2xN) return indices along the x, then the y direction

        indices = (point - self.lower_corner)/ self.world_shape_vect * self.map_shape_vect      
        return indices.astype(int)      

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        # print("TO DO: Implement a method to get the pixel locations of the robot path")

        # Responsible: Ben
        # input: (2xN) assume to be center of the robot in world coordinate
        # Output: 2 arrays: 1) x-indices of pixels occupied by the robot at all points 2) corresponding y-indices of pixels occupied by the robot at all points
        footprints_idx_x = []
        footprints_idx_y = []
        pixel_centers = self.point_to_cell(points)      #convert to pixel center, then obtain footprint -> less conversion

        for i in range(points.shape[1]):
            new_idx_x, new_idx_y = circle(pixel_centers[0,i], pixel_centers[1,i], self.robot_pixel_radius) 
            footprints_idx_x.extend(new_idx_x)
            footprints_idx_y.extend(new_idx_y)

        return footprints_idx_x, footprints_idx_y
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point

        # Responsible: Ben

        # convert to node_i's frame for easier velocities and trajectory rollout calculation
        theta_i_w = node_i[2, 0]
        T_w_i = np.array([[np.math.cos(theta_i_w), -np.math.sin(theta_i_w), node_i[0,0]],
                          [np.math.sin(theta_i_w),  np.math.cos(theta_i_w), node_i[1,0]],
                          [0, 0, 1]])    #ece470 convention
        T_i_w = np.linalg.inv(T_w_i)

        node_i_i = np.zeros((3, 1))
        point_f_i_aug = np.matmul(T_i_w,  np.vstack((point_f, 1)))
        point_f_i = point_f_i_aug[:2]

        vel, rot_vel = self.robot_controller_exact(node_i_i, point_f_i)
        robot_traj_i = self.trajectory_rollout(vel, rot_vel)
        
        # convert back to world coordinate for collision detection
        robot_traj_pts = np.matmul(T_w_i, np.vstack((robot_traj_i[:2, :], np.ones((1, self.num_substeps)))))      # convert x,y coord from frame i to world frame
        if self.traj_has_collision(robot_traj_pts):
            return False, None
        else:
            #no collision -> calculate the correct heading, and return
            
            robot_traj_pts[2, :] = (theta_i_w + robot_traj_pts[2, :]) % np.math.pi
            return True, robot_traj_pts


        # Ben: i dont think we can just return the trajectory, because trajectory may not exist
        # print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        # return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        #print("TO DO: Implement a cost to come metric")

        traj = trajectory_o
        eudist = 0 #euclidean distance
        
        for i in range(0, traj.shape[1]-1): #iterate through number of columns
            eudist += np.linalg.norm(traj[:,i][0:2] - traj[:,i+1][0:2]) #calculate euclidean norm between trajectory points
        
        #print('eudist')
        return eudist
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        #print("TO DO: Update the costs of connected nodes after rewiring.")
        
        #can use a queue for Breadth first search traversal
        #source: https://www.educative.io/edpresso/how-to-implement-a-breadth-first-search-in-python

        #initialize queue of node ids
        queue = np.array([])

        #redundant visited node list
        visited = np.array([])

        #push node_id given
        queue = np.append(queue,node_id)
        visited = np.append(visited,node_id)

        #queue list isnt empty (visited all the children nodes)
        while queue.size != 0:
            #take explored nodeID as an Int
            nodeID = int(queue[0])

            print(queue.size)
            print(nodeID)
            #pop explored node
            queue = queue[1:]

            for childID in self.nodes[nodeID].children_ids:
                if childID not in visited: #redundancy
                    #push node to visitied and queue list
                    visited = np.append(visited,childID)
                    queue = np.append(queue,childID)

                    #update cost of children with euclidean distance
                    dist = np.linalg.norm(self.nodes[nodeID].point[0:2]-self.nodes[childID].point[0:2])
                    self.nodes[childID].cost = self.nodes[nodeID].cost + dist
        
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        
        # responsible: Ben
        for i in range(self.max_rrt_iteration): #Most likely need more iterations than this to complete the map!
            #Sample map space
            if i > self.box_shrink_start:       # shrink the box at shrink_ratio after box_shrink_start number of iterations
                self.box_ratio = self.final_box_ratio  + self.shrink_ratio * (self.box_ratio - self.final_box_ratio )

            # use 1 of three kinds of pont samples: goal point, goal region, and shrinking total map region
            if np.random.random() > self.goal_sample_rate + i * self.goal_sample_rate_inc:      # want to bias the sampler towards the goal
                if np.random.random() > self.goal_region_sample_rate:
                    point = self.sample_map_space()                 # shrinking total map region, for random exploration (before box_shrink_start), and focus exploitation (after box_shrink_start)
                else:
                    point = self.sample_goal_region_space()         # goal region -> ensure explore regions around the goal more -> bias towards the goal region
            else:
                point = self.goal_point                             # sample goal_point exactly to prompt exact solution to the goal

            if i > self.box_shrink_start:
               self.sampled_pts.append(point)
                
            #Get the closest point
            closest_node_id = self.closest_node(point)       

            #Simulate driving the robot towards the closest point
            traj_exist, trajectory_o, trans_vel = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            
            self.closest_pts[:, [i]] = self.nodes[closest_node_id].point
            if (not traj_exist):
                # save locations that are not promising (no traj possible) -> dont explore this region again
                closest_point = self.nodes[closest_node_id].point

                self.node_dist_scaling[closest_node_id] = np.inf            # set distance to these points to be infinite -> will not be identified as closest points to subsequent point samples
                invalid_point_cell = self.point_to_cell(closest_point[:2, [0]])
                self.invalid_locations[invalid_point_cell[1,0], invalid_point_cell[0,0]] = 1
                self.invalid_pts += 1

                ## Note: rather than storing the invalid points, may also store all points within a certain radius of the invalid point. In my experience, not very useful
                # d_sq_list = [((self.nodes[i].point[0,0] - closest_point[0,0])**2 + (self.nodes[i].point[1,0] - closest_point[1,0])**2)*self.node_dist_scaling[i]
                #     for i in range(len(self.nodes))]
                # invalid_inds = np.where(np.array(d_sq_list) < self.invalid_radius)[0]
                # for i in invalid_inds:
                #     self.node_dist_scaling[i] = np.inf
                #     invalid_point_cell = self.point_to_cell(self.nodes[i].point[:2, [0]])
                #     self.invalid_locations[invalid_point_cell[1,0], invalid_point_cell[0,0]] = 1
                # self.invalid_pts += len(invalid_inds)
                # print(self.invalid_pts)
                continue
            
            else:
                # add the final point of the trajectory
                parent_id = closest_node_id
                
                cost = trans_vel * self.traj_time #old cost, Needs to be fixed
                
                #cost = self.cost_to_come(trajectory_o) #update cost to come using trajectory? - AK WROMG
                
                new_node = trajectory_o[:, [-1]]
                new_node_cell = self.point_to_cell(new_node[:2,[0]])

                # only add nodes that are not duplicated, this method is much faster (though less accurate) than the check_if_duplicate function
                if self.visitted_locations[new_node_cell[1,0], new_node_cell[0,0]] == 0:
                    # not yet visitted
                    
                    self.nodes[parent_id].children_ids.append(len(self.nodes)) #update children ids - AK
                    
                    self.nodes.append(Node(new_node, parent_id, cost))
                    self.node_dist_scaling.append(1)
                    self.visitted_locations[new_node_cell[1,0], new_node_cell[0,0]] = 1
                else:
                    continue
                
                # check if has reached the goal
                traj_deviation_from_goal = trajectory_o[:2,:] - self.goal_point
                traj_dist_from_goal_sq = traj_deviation_from_goal[0,:]**2 + traj_deviation_from_goal[1,:]**2
                if any(traj_dist_from_goal_sq < self.stopping_dist_sq):
                    # a point in the trajectory is at the goal location
                    minind = argmin(traj_dist_from_goal_sq)
                    self.nodes.append(Node(trajectory_o[:, [minind]], parent_id, cost))
                    self.node_dist_scaling.append(1)
                    print('reached goal \n\n\n\n')
                    break


            #Check for collisions
            # print("TO DO: Check for collisions and add safe points to list of nodes.")
            
            #Check if goal has been reached
            # print("TO DO: Check if at goal point.")
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

    def plot_stuff(self):
        x_all = np.array([node.point[0,:] for node in self.nodes]) 
        y_all = np.array([node.point[1,:] for node in self.nodes])
        node_points_all = np.hstack((x_all, y_all)).T
        node_cells_all = self.points_to_robot_circle(node_points_all)
        node_centers_all = self.point_to_cell(node_points_all)
        self.axs[0].scatter(node_centers_all[0,:], node_centers_all[1,:])
        goal_point_cell = self.point_to_cell(self.goal_point)
        self.axs[0].scatter(goal_point_cell[0,:], goal_point_cell[1.:])

        clostest_cells = self.point_to_cell(self.closest_pts[:2, :])
        self.axs[0].scatter(clostest_cells[0,:], clostest_cells[1.:])

        invalid_idx = np.where(np.array(self.node_dist_scaling) == np.inf)[0]
        self.axs[0].scatter(node_centers_all[0,invalid_idx], node_centers_all[1,invalid_idx])


        if len(self.sampled_pts) > 0:
            self.axs[1].imshow(self.occupancy_map)
            sampled_pts = np.array(self.sampled_pts).squeeze(2).T
            sampled_cells = self.point_to_cell(sampled_pts[:2, :])
            self.axs[1].scatter(sampled_cells[0,:], sampled_cells[1.:])

        print(self.invalid_pts)
        plt.show()

def main():
    #Set map information
    # map_filename = "simple_map.png"
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    goal_point = np.array([[42], [25]]) #m      # Ben: real goal
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    
    #### part 1 unit test
    # test_array = np.array([ [0, 1], [1, 2] ])
    # print(path_planner.points_to_robot_circle(test_array))    
    
    #### part 2 unit test
    node_i = np.array([[2], [1], [np.math.pi/2]])
    point_s = np.array([[0], [3]])
    path_planner.simulate_trajectory(node_i, point_s)
    
    #### part 3 unit test
    # path_planner.nodes.append(Node(np.array([[1], [0], [0]]), -1, 0))
    # path_planner.nodes.append(Node(np.array([[0], [1], [0]]), -1, 0))
    # print(path_planner.check_if_duplicate(np.array([[0], [2]])))
    # print(path_planner.check_if_duplicate(np.array([[0], [1]])))
    # print(path_planner.closest_node(np.array([[0], [1]])))
    # print(path_planner.closest_node(np.array([[1.5], [0]])))


    start = time.time()
    nodes = path_planner.rrt_planning()
    print(time.time()-start)
    # path_planner.plot_stuff()
    # time.sleep(1)


    # nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())
    plt.figure()
    x = node_path_metric[0, :]
    y = node_path_metric[1, :]
    node_cells = path_planner.point_to_cell(node_path_metric[:2, :])
    plt.imshow(path_planner.occupancy_map)
    # plt.plot(node_cells[0,:], node_cells[1,:])
    plt.plot(node_cells[0,:], node_cells[1,:])
    plt.show()
    time.sleep(1)


    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
