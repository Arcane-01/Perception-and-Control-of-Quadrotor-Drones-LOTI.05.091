import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.random import PRNGKey, split, normal
from functools import partial
import time
import matplotlib.pyplot as plt

class RandomSamplingPlanner():

	def __init__(self):
		pass

	def compute_controls(self, key, linear_velocity, position_setpoint, obstacles):

		# linear_velocity (3,)
		# position_setpoint (3,)
		# obstacles (L_max_lidar,3)

		pass

		# return  vel_optimal, best_traj_plan, sampled_trajs_plan
		
		## vel_optimal (3,), sampled_trajs_plan (N_SAMPLESxN_STEPSx3), best_traj_plan (N_STEPSx3)