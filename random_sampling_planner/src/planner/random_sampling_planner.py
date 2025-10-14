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