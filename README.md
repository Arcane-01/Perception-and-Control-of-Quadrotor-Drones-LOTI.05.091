# Perception and Control of Quadrotor Drones (LOTI.05.091)

Repository associated with the course **Perception and Control of Quadrotor Drones (LOTI.05.091)**.

---
## Installation

**Note:** This code has only been tested with ROS-Noetic on Ubuntu 20.04. It is assumed that you already have a complete ROS installation.

### 1. Install required packages

```bash
sudo apt-get install libavahi-client-dev
sudo apt install ros-noetic-message-to-tf
sudo apt install ros-noetic-twist-mux
```



### 2. Create a new catkin workspace

```bash
cd
mkdir -p bebop_hardware_ws/src
cd bebop_hardware_ws
catkin init
```

### 3. Clone the repository

```bash
cd bebop_hardware_ws/src
git clone https://github.com/Arcane-01/Perception-and-Control-of-Quadrotor-Drones-LOTI.05.091.git -b bebop_hardware .
```

### 4. Build the workspace

```bash
cd ~/bebop_hardware_ws
catkin build
source devel/setup.bash
```

### 5. Set up environment variable

Add the following line to your `.bashrc` (replace `path to your bebop_hardware_ws` with the actual path):

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path_to_your_bebop_hardware_ws/devel/lib/parrot_arsdk
``` 

# **Usage Instructions**

### **1. Connect to motion capture (mocap) system via Ethernet and then run the VRPN client launch file**

```bash
roslaunch vrpn_client_ros sample.launch server:=192.168.1.134
```

### **2. Connect to Bebop via Wi-Fi and then launch the driver**

```bash
roslaunch bebop_driver bebop_with_vel_controller.launch
```

### **3. Take off**

```bash
rostopic pub /bebop/takeoff std_msgs/Empty "{}"
```

**IMPORTANT:** Keep the following ready in a new terminal for emergency landing:

```bash
rostopic pub /bebop/land std_msgs/Empty "{}"
```

### **4. Get a dummy point cloud**

```bash
roslaunch dummy_pcd dummy_pcd_create.launch
```

### **5. Run the random sampling planner with RViz**

```bash
roslaunch random_sampling_planner bebop_nav.launch
```

##  NOTE:
-  The dummy point clouds you get are in the local frame (same as simulation). Make sure that obstacle distance computations in all your codes use the drone pose as (0,0).

- The necessary velocity clipping and transformations to account for small variations in the yaw angle of the Bebop are already implemented in `quadnav.py`.

- Finite differencing of the motion capture pose is used to compute the velocity.

-  The goal position is hardcoded in `quadnav.py` and can be changed directly from there.
