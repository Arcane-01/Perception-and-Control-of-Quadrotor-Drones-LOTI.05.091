import numpy as np

from PyFlyt.core import Aviary
from PyFlyt.core.abstractions import ControlClass
import pybullet as p

class CustomController(ControlClass):

	def __init__(self):

		# ---------- LIDAR  ----------
		self.num_rays = 360
		ray_length = 10.0
		ray_start_offset = 0  	#  offset from drone center
		self.local_ray_from = []
		self.local_ray_to = []
		self.theta_list = []

		for i in range(self.num_rays):

			theta = 2 * np.pi * float(i) /self.num_rays  
			self.theta_list.append(theta)
			x = np.sin(theta)
			y = np.cos(theta)
			self.local_ray_from.append([ray_start_offset*x, ray_start_offset*y, 0])
			self.local_ray_to.append([ray_length*x, ray_length*y, 0])
		# -----------------------------
	
	def pyflyt_lidar(self):

		drone_id = 1  
		link_idx = -1
		results = p.rayTestBatch(self.local_ray_from, self.local_ray_to, parentObjectUniqueId=drone_id, parentLinkIndex=link_idx)
		distances = np.zeros(self.num_rays)
		
		for i, res in enumerate(results):
			hit_fraction = res[2]
			hit_pos = [
				self.local_ray_from[i][0] + hit_fraction*(self.local_ray_to[i][0] - self.local_ray_from[i][0]),
				self.local_ray_from[i][1] + hit_fraction*(self.local_ray_to[i][1] - self.local_ray_from[i][1]),
				self.local_ray_from[i][2] + hit_fraction*(self.local_ray_to[i][2] - self.local_ray_from[i][2])
			]
			# color = [0, 1, 0] if hit_fraction >= 1.0 else [1, 0, 0]
			# p.addUserDebugLine(self.local_ray_from[i], hit_pos, color, lineWidth=1, lifeTime=5.0,
			# 				parentObjectUniqueId=drone_id, parentLinkIndex=link_idx)
			distances[i] = np.linalg.norm([hit_pos[0]-self.local_ray_from[i][0], hit_pos[1]-self.local_ray_from[i][1]])
		
		hit_points = []
		drone_local_pos = np.array([0.0, 0.0])
		for d, t, res in zip(distances, self.theta_list, results):
			hit_fraction = res[2]
			if hit_fraction < 1.0:  
				x = drone_local_pos[0] + d*np.sin(t)
				y = drone_local_pos[1] + d*np.cos(t)
				hit_points.append([x, y])

		hit_points = np.array(hit_points)

		return hit_points

	def reset(self):
		pass
	
	def step(self, state: np.ndarray, setpoint: np.ndarray):
		target_velocity = self.apf(observation=state, position_setpoint=np.array([*setpoint[:2], setpoint[-1]]))
		target_yaw_rate = 0.0
		output = np.array([*target_velocity[:2], target_yaw_rate, target_velocity[-1]])
		return output

	def apf(self, observation, position_setpoint):

		angular_velocity = observation[0] # shape:(3,) ## body frame angular velocity
		angular_position = observation[1] # shape:(3,) ## ground frame angular position
		linear_velocity = observation[2]  # shape:(3,) ## body frame linear velocity
		linear_position = observation[3]  # shape:(3,) ## ground frame linear position
		goal = position_setpoint 		  # shape:(3,) ## (x_goal, y_goal, z_goal)
		obstacles = self.pyflyt_lidar()	  # shape:(N, 2) 
		
		k_att = .5
		u_att = k_att * (goal - linear_position)

		v = u_att

		return np.array(v)

# the starting position and orientation
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(
	start_pos=start_pos, 
	start_orn=start_orn, 
	render=True, 
	drone_type="quadx"
	)

env.drones[0].register_controller(
	controller_constructor=CustomController, controller_id=8, base_mode=5
)
env.set_mode(8)
setpoint = np.array([1.0, 2.0, 0.0, 2.0]) # (x, y, yaw, z)
env.set_setpoint(0, setpoint)

# Load obstacles
obstacles = [
	[0,2,0.5],
	[0,-2,0.5],
	[2,0,0.5],
	[-2,0,0.5]
]
for pos in obstacles:
	env.loadURDF("./obstacles/cylinder.urdf", basePosition=pos, useFixedBase=True)
env.register_all_new_bodies()

# run the sim
for i in range(1000):
	env.step()