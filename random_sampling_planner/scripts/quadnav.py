#!/usr/bin/env python3.10

import rospy
import numpy as np
import jax
import random

from tf import transformations
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Twist, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray

import open3d as o3d
import open3d_conversions

from scipy.spatial.transform import Rotation as R

from planner.random_sampling_planner import RandomSamplingPlanner

class QuadNav:
	def __init__(self):
		rospy.init_node("quadrotor_navigation")
		print("[QUADROTOR NAVIGATION] Initializing the 'quadrotor_navigation' Node...")

		#----------------------------------------------------------------------------------

		## Planner

		self.planner = RandomSamplingPlanner() 
		self.key = jax.random.PRNGKey(0)
		self.num = 0
		#----------------------------------------------------------------------------------

		# Observation topics
		self.cloud_topic = rospy.get_param('~cloud_topic','/pointcloud')
		self.odom_topic = rospy.get_param('~odom_topic','/bebop/odom')

		# Publishers
		self._vel_pub = rospy.Publisher("/cmd_vel", Twist,queue_size=10)
		
		# RViZ Visualization publishers
		self._markerarr_pub = rospy.Publisher('/primitives_ellite', MarkerArray, queue_size=10)
		self._linestrip_pub = rospy.Publisher('/trajectory_optimal', Marker, queue_size=10)
		self._arrowmarker_pub = rospy.Publisher('/goal_marker', Marker, queue_size=10)
		self._spheremarker_pub = rospy.Publisher('/goaltf_marker', Marker, queue_size=10)

		#----------------------------------------------------------------------------------

		# Parameters 

		# PointCloud Downsampling
		self.L_max_lidar = 300

		# class variables for storing observation data
		self.odom = None
		self.goal = None
		self.transformed_goal = None
		self.rotation_mtx = None

		# Odometry
		self.pose = None
		self.linear_vel = None
		self.linear_acc = None
		self.prev_linear_vel = np.array([0.0, 0.0, 0.0])
		self.prev_time = None
		self.prev_yaw_e = 0.0

		# PCD
		self.cloud = None
		self.min_dist = None

		#----------------------------------------------------------------------------------

		# ############# Goal position ###################
		self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.handle_goal)

		#----------------------------------------------------------------------------------

		# Observation subscribers
		odometry_sub = Subscriber(self.odom_topic, Odometry) 
		lidar_pcd_sub = Subscriber(self.cloud_topic, PointCloud2)

		ats = ApproximateTimeSynchronizer([odometry_sub, lidar_pcd_sub], queue_size=10, slop=0.1)
		ats.registerCallback(self.ats_callback)

		#----------------------------------------------------------------------------------

		# ROS Timer
		self.config_planner_frequency = 15       # Planner called every 0.066 seconds
		self.timer_save = rospy.Timer(rospy.Duration(1. / self.config_planner_frequency),
									self._call_planner)

		#----------------------------------------------------------------------------------

		print("[QUADROTOR NAVIGATION] Initialization completed!")

	def _call_planner(self, event):
		if (self.odom is not None) and \
			(self.goal is not None) and \
			(self.transformed_goal is not None) and \
			(self.cloud is not None) :

			key, _ = jax.random.split(self.key)
			
			vel_optimal, best_traj_plan,sampled_trajs_plan = self.planner.compute_controls(key = key, 
																							linear_velocity = self.linear_vel,
																							position_setpoint =  self.transformed_goal,
																							obstacles = self.cloud)

			self.key = key

			if vel_optimal != None and np.linalg.norm(self.pose[:3] - self.goal_arr[:3]) > 0.5:
				self.publish_cmd_vel_msg(vel_optimal) ## vel_optimal (3,)

			if np.linalg.norm(self.pose[:3] - self.goal_arr[:3]) < 0.5:
				print("!!REACHED GOAL!!")

			self._visualize_trajectories(sampled_trajs_plan, best_traj_plan) ## sampled_trajs_plan (N_SAMPLESxN_STEPSx3), best_traj_plan (N_STEPSx3)
			self._visualize_goal()
			self._visualize_goal_tf(self.transformed_goal)

			self.num += 1

	def handle_goal(self, goal_data):
		self.goal = goal_data
		self.goal.pose.position.z = random.uniform(2.5,5.0)
		self.goal_arr = np.array([self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z])
		self.num = 0

	def ats_callback(self, odom_data, lidar_pcd_ros):
		
		if lidar_pcd_ros is None or \
			odom_data is None:
			return
		#------------------------------------------

		## Odometry Processing

		self.odom = odom_data

		quaternion = (
			odom_data.pose.pose.orientation.x,
			odom_data.pose.pose.orientation.y,
			odom_data.pose.pose.orientation.z,
			odom_data.pose.pose.orientation.w
		)
		roll, pitch, yaw = transformations.euler_from_quaternion(quaternion)
		
		self.pose = np.array([
			odom_data.pose.pose.position.x, 
			odom_data.pose.pose.position.y, 
			odom_data.pose.pose.position.z,
			roll,
			pitch,
			yaw])

		linear_vel_odom = np.array([
			odom_data.twist.twist.linear.x,
			odom_data.twist.twist.linear.y,
			odom_data.twist.twist.linear.z
		])
		
		self.rotation_mtx = R.from_quat(np.array([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w])).as_matrix()
		
		if self.goal is not None:    
			self.transformed_goal = self.rotation_mtx.T @ (self.goal_arr - np.array([self.pose[0], self.pose[1], self.pose[2]]))
		
		self.linear_vel = self.rotation_mtx.T @ linear_vel_odom

		# computing acceleration
		current_time = rospy.Time.now().to_sec()
		
		if self.prev_time is not None:
			self.linear_acc = (self.linear_vel - self.prev_linear_vel)/(current_time - self.prev_time + (1e-10))

		self.prev_time = current_time
		self.prev_linear_vel = self.linear_vel
		#------------------------------------------
		
		## PCD Processing

		_lidar_o3d_pcd = open3d_conversions.from_msg(lidar_pcd_ros)                             # ROS PointCloud2 msgs to Open3D PointClouds

		_lidarpoints = np.asarray(_lidar_o3d_pcd.points)

		lidar_o3d_pcd = o3d.geometry.PointCloud()
		lidar_o3d_pcd.points = o3d.utility.Vector3dVector(_lidarpoints)

		if np.asarray(lidar_o3d_pcd.points).shape[0] > self.L_max_lidar:
			lidar_o3d_pcd = lidar_o3d_pcd.farthest_point_down_sample(self.L_max_lidar) # Downsample the point clouds

		if lidar_o3d_pcd.is_empty():
			obs_pt = np.array([1e10, 1e10, 1e10])
			obs_pt = np.reshape(obs_pt, (1,3))
			lidar_o3d_pcd.points.extend(o3d.utility.Vector3dVector(obs_pt))

		lidar_pcd_ds = np.asarray(lidar_o3d_pcd.points)

		if lidar_pcd_ds.shape[0] < self.L_max_lidar:
			lidar_pcd_ds = np.vstack([lidar_pcd_ds, np.tile(lidar_pcd_ds[-1], (self.L_max_lidar - lidar_pcd_ds.shape[0], 1))]) # appending the last point          

		self.cloud = lidar_pcd_ds

	def compute_motion_cmd(self, vel):

		cmd = Twist()
		vel_x = vel[0]
		vel_y = vel[1]
		vel_z = vel[2]

	
		cmd.linear.x = vel_x 
		cmd.linear.y = vel_y
		cmd.linear.z = vel_z
	
		return cmd

	def publish_cmd_vel_msg(self, vel):
		try:
			cmd = self.compute_motion_cmd(vel)
			self._vel_pub.publish(cmd)
		except:
			print("")

	def _create_primitive_marker(self, idx, data, namespace, rgb):

		marker = Marker()
		marker.type = Marker.LINE_STRIP
		marker.action = Marker.ADD
		marker.scale = Vector3(0.05, 0.01, 0)          # only scale.x used for line strip
		marker.color.r = rgb[0]                        # color 
		marker.color.g = rgb[1]                        
		marker.color.b = rgb[2]                         
		marker.color.a = 1.0                           # alpha - transparency parameter

		marker.ns = namespace
		marker.pose.orientation.w = 1.0

		for i in range(data.shape[0]):
			point = Point()
			point.x = data[i, 0]
			point.y = data[i, 1]
			point.z = data[i, 2]
			marker.points.append(point)

		marker.id = idx
		marker.header.stamp = rospy.get_rostime()
		marker.lifetime = rospy.Duration(0.0667)
		marker.header.frame_id = "base_link"

		return marker

	def _visualize_goal(self):

		if self.goal is None:
			return
		
		# scale.x is the arrow length, scale.y is the arrow width and scale.z is the arrow height.
		marker = Marker()
		marker.type = Marker.ARROW
		marker.action = Marker.ADD
		marker.scale = Vector3(1.0, 0.075, 0.1)          
		marker.color.r = 1.0                               
		marker.color.g = 1.0                        
		marker.color.b = 0.0                         
		marker.color.a = 1.0                           

		marker.ns = "goal_marker"

		marker.pose.position.x = self.goal.pose.position.x
		marker.pose.position.y = self.goal.pose.position.y
		marker.pose.position.z = self.goal.pose.position.z
		marker.pose.orientation.x = self.goal.pose.orientation.x
		marker.pose.orientation.y = self.goal.pose.orientation.y
		marker.pose.orientation.z = self.goal.pose.orientation.z
		marker.pose.orientation.w = self.goal.pose.orientation.w

		marker.id = 0
		marker.header.stamp = rospy.get_rostime()
		marker.lifetime = rospy.Duration(0.0)
		marker.header.frame_id = "odom"

		self._arrowmarker_pub.publish(marker)

	def _visualize_goal_tf(self, transformed_goal):

		if self.goal is None:
			return
		
		marker = Marker()
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.scale = Vector3(0.1, 0.1, 0.1)              # sphere scaling
		marker.color.r = 0.0                               # color 
		marker.color.g = 0.0                       
		marker.color.b = 0.5                         
		marker.color.a = 1.0                               # alpha - transparency parameter

		marker.ns = "goaltf_marker"

		marker.pose.position.x = transformed_goal[0]
		marker.pose.position.y = transformed_goal[1]
		marker.pose.position.z = transformed_goal[2]

		marker.id = 0
		marker.header.stamp = rospy.get_rostime()
		marker.lifetime = rospy.Duration(0.0)
		marker.header.frame_id = "base_link"

		self._spheremarker_pub.publish(marker)

	def _visualize_trajectories(self, primitives_ellite, traj_optimal):
		
		marker_array = MarkerArray()
		for idx in range(len(primitives_ellite)):
			marker = self._create_primitive_marker(idx, primitives_ellite[idx], "primitives_ellite", [0.0, 1.0, 0.0])
			marker_array.markers.append(marker)
		
		trajoptimal_marker = self._create_primitive_marker( 0, traj_optimal, "traj_optimal", [1.0, 0.0, 0.0])

		self._markerarr_pub.publish(marker_array)
		self._linestrip_pub.publish(trajoptimal_marker)


if __name__ == '__main__':
	quadrotor_navigation = QuadNav()
	rospy.spin()