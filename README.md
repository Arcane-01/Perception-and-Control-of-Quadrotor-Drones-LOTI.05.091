# cbf-obstacle-avoidance
Control Barrier Function(CBF) implementation for quadrotor navigation

We use the standard CBF QP formulation,

$$
\begin{align*}
\text{minimize} \quad & \frac{1}{2} \|v - v_{\text{des}}\|^2 \\
\text{subject to} \quad & \nabla h_i(x)^T v \geq -\alpha h_i(x) \quad \forall i \in \text{obstacles}
\end{align*}
$$

where, 
$h_i(x)$ is the distance to obstacle $i$

In this repository, this QP is solved using CVXOPT.

## Results
#### Navigation in Pillar Worlds
<p align="center">
<img src="media/cbf_env2.gif">
</p>
<p align="center">
<img src="media/cbf_env4.gif">
</p>

#### Navigation in Sphere Worlds
<p align="center">
<img src="media/cbf_spheres6.gif">
</p>

## Effect of varying the parameter $\alpha$

The parameter $\alpha$ in CBF significantly affects how aggressively the robot avoids obstacles
<!--
<p align="center">
  <img src="media/alpha_0,1.gif" width="20%" alt="alpha 0.1">&nbsp;&nbsp;&nbsp;
  <img src="media/alpha_1.gif" width="20%" alt="alpha 1.0">&nbsp;&nbsp;&nbsp;
  <img src="media/alpha_10.gif" width="20%" alt="alpha 10.0">
</p>

<p align="center">
  α = 0.1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  α = 1.0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  α = 10.0
</p> -->


<table align="center" cellpadding="10" border="0.001">
   <tr>
    <td align="center">
      <img src="media/alpha_0,1.gif" width="250px"><br>
      α = 0.1
    </td>
    <td align="center">
      <img src="media/alpha_1.gif" width="250px"><br>
      α = 1.0
    </td>
    <td align="center">
      <img src="media/alpha_10.gif" width="250px"><br>
      α = 10.0
    </td>
  </tr>
</table> 



## Installation

Note that this code has only been tested with ROS-Noetic on Ubuntu 20.04. It is assumed that you already have a complete ROS installation. Also, ensure that CVXOPT is installed.

1. (optional) Set up a catkin workspace

	```
	cd
	mkdir -p catkin_ws/src
	cd catkin_ws
	catkin init
	```

2. Clone this repository in the workspace 

	```
	cd catkin_ws/src
	git clone https://github.com/Arcane-01/cbf-quadrotor-navigation.git .
	```

3. Build the workspace

	```
	cd ../
	catkin build
	source devel/setup.bash
	```

## Usage

1. To launch the Gazebo simulation with Bebop and CBF node, run the following command:
	```
	bash run_cbf_quadrotor_navigation.sh <world_name> 
	```
 

	* `<world_name>`: Gazebo world file from [`cbf/worlds`](cbf/worlds) (defaults to env4)

2. Once the CBF node has been initialized, you can begin navigation by setting a 2D Nav Goal in RViZ.

## References

- *Singletary et al.*, “Comparative Analysis of Control Barrier Functions and Artificial Potential Fields for Obstacle Avoidance” [arXiv](https://arxiv.org/abs/2010.09819)
