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

		self.DT = 0.1             # Time step
		self.N_STEPS = 50         # Number of time steps in trajectory (horizon)
		self.N_SAMPLES = 1000     # Number of random trajectories to sample (can be much larger with JAX)

		# System Dynamics (Double Integrator)
		self.STATE_DIM = 4
		self.CONTROL_DIM = 2

		# Control Sampling Parameters
		self.CONTROL_STD_DEV = .5

		# Cost Function Weights
		self.W_GOAL_P = 10.0
		self.W_GOAL_V = 1.0
		self.W_CONTROL = 0.01
		self.W_OBSTACLE = 100.0
	
		self.ROBOT_RADIUS = 0.25
		self.D_MARGIN = 0.5
		self.A_d, self.B_d = self.get_system_matrices(self.DT)

		self.control_mean = jnp.zeros(( self.N_STEPS, self.CONTROL_DIM  ))
		
		# Vmapping
		self.vmapped_cost = vmap(self.calculate_cost, in_axes=(0, 0, None, None))
		self.vmapped_rollout = vmap(self.rollout_trajectory, in_axes=(None, 0))
	
	@partial(jit, static_argnums=(0,)) 
	def get_system_matrices(self, dt):

		state_dim = self.STATE_DIM
		control_dim = self.CONTROL_DIM
		
		A_c = jnp.zeros((state_dim, state_dim)).at[jnp.arange(state_dim//2), jnp.arange(state_dim//2, state_dim)].set(1)
		B_c = jnp.zeros((state_dim, control_dim)).at[jnp.arange(state_dim//2, state_dim), jnp.arange(control_dim)].set(1)
		A_d = jnp.eye(state_dim) + A_c * dt
		B_d = B_c * dt
		return A_d, B_d

	@partial(jit, static_argnums=(0,)) 
	def rollout_trajectory(self, start_state, controls):
		def step(state, control):
			next_state = self.A_d @ state + self.B_d @ control
			return next_state, next_state

		_, trajectory = lax.scan(step, start_state, controls)
		return jnp.vstack([start_state, trajectory])

	@partial(jit, static_argnums=(0,)) 
	def calculate_cost(self, trajectory, controls, goal_state, obstacles):

		final_state = trajectory[-1]
		error = final_state - goal_state
		
		Q = jnp.diag(jnp.array([self.W_GOAL_P, self.W_GOAL_P, self.W_GOAL_V, self.W_GOAL_V]))
		goal_cost = error.T @ Q @ error

		control_cost = jnp.sum(controls**2) * self.W_CONTROL

		pos = trajectory[:, :2]  

		dist_to_centers = jnp.linalg.norm(pos[:, None, :] - obstacles[None, :, :2], axis=2)

		dist = dist_to_centers - obstacles[:, 2] - self.ROBOT_RADIUS

		cost_per_point = self.W_OBSTACLE * (1.0 / dist - 1.0 / self.D_MARGIN)**2
		point_wise_cost = jnp.where(dist < self.D_MARGIN, cost_per_point, 0.0)
		point_wise_cost = jnp.where(dist <= 0, jnp.inf, point_wise_cost)

		obstacle_cost = jnp.sum(jnp.sum(point_wise_cost, axis=1))

		return goal_cost + control_cost + obstacle_cost

	@partial(jit, static_argnums=(0,5))
	def plan_step(self, key, current_state, goal_state, obstacles, n_samples, control_mean):

		controls_shape = (n_samples, self.N_STEPS, self.CONTROL_DIM)
		all_controls = normal(key, shape=controls_shape) * self.CONTROL_STD_DEV+control_mean


		all_trajectories = self.vmapped_rollout(current_state, all_controls)


		all_costs = self.vmapped_cost(all_trajectories, all_controls, goal_state, obstacles)


		best_idx = jnp.argmin(all_costs)
		best_controls = all_controls[best_idx]

		idx_cost_sort = jnp.argsort(all_costs)
		all_trajectories = all_trajectories[idx_cost_sort]

		return best_controls[0], all_trajectories[best_idx], all_trajectories[:500], best_controls
	
	@partial(jit, static_argnums=(0,)) 
	def compute_controls(self, key, linear_velocity, position_setpoint, obstacles):

		goal = position_setpoint 		  # shape:(3,) ## (x_goal, y_goal, z_goal)

		obs_radius = 0.15                  
		obstacles = jnp.concatenate([obstacles, jnp.full((obstacles.shape[0], 1), obs_radius)], axis=1)		

		current_state = jnp.array([0.,0.,*linear_velocity[:2]])
		goal_state = jnp.array([*goal[:2], 0., 0.])

		best_control, best_traj_plan, sampled_trajs_plan, best_control_full = self.plan_step(
            key, current_state, goal_state, obstacles, self.N_SAMPLES, self.control_mean
        )
		 
		proposed_v = linear_velocity[:2] + best_control * (0.35)

		v = proposed_v

		return v, best_traj_plan,sampled_trajs_plan