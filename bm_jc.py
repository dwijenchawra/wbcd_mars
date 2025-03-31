#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
import numpy as np
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions
from pxr import PhysxSchema
import threading
import subprocess
import json
import logging
import torch
from isaaclab.utils.math import quat_from_matrix
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
import time
import asyncio
import sys


# Open the stage
open_stage(usd_path="/home/dwijen/Documents/CODE/IsaacLab/wbcd/bimanual_warehouse/World0.usd")
world = World()

# Get handles to both UR10e robots and grippers
arm_left_prim_path = "/World/ur10e_robotiq2f_140"
arm_right_prim_path = "/World/ur10e_robotiq2f_141"

arm_left = Articulation(arm_left_prim_path, name="arm_left")
arm_right = Articulation(arm_right_prim_path, name="arm_right")

# Reset and initialize world
world.reset()

# Initialize the articulations
arm_left.initialize()
arm_right.initialize()

stage = world.stage

#increase gpu found lost pairs capacity
scenePrim = stage.GetPrimAtPath("/World/ur10e_robotiq2f_140/PhysicsScene")
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scenePrim)
physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(10 * 1024)

scenePrim = stage.GetPrimAtPath("/World/ur10e_robotiq2f_141/PhysicsScene")
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scenePrim)
physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(10 * 1024)


# # print all fields for these articulations
# print("[INFO]: Printing joint names for both arms...")
# print(arm_left.joint_names)
# print(arm_right.joint_names)


# Initial joint positions (in radians) for a reasonable starting pose
# Format: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
left_joint_positions = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])
right_joint_positions = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])

# Noise parameters
noise_amplitude = 0.05  # Maximum noise in radians (about 3 degrees)

# Create initial ArticulationAction for position control
left_action = ArticulationActions(
    joint_names=None,
    joint_indices=np.arange(6),
    joint_positions=left_joint_positions,
    joint_velocities=None,
    joint_efforts=None
)

right_action = ArticulationActions(
    joint_names=None,
    joint_indices=np.arange(6),
    joint_positions=right_joint_positions,
    joint_velocities=None,
    joint_efforts=None
)
frame_marker_cfg = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"/home/dwijen/Documents/CODE/IsaacLab/wbcd/assets/Collected_frame_prim/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        )
    }
)
# Marker configurations for visualization
frame_marker_cfg = VisualizationMarkersCfg(
    markers={
        "hand_plane": sim_utils.UsdFileCfg(
            usd_path=f"/home/dwijen/Documents/CODE/IsaacLab/wbcd/assets/Collected_frame_prim/frame_prim.usd",
            scale=(0.2, 0.2, 0.2),  # Slightly larger scale for visibility
        )
    }
)
hand_plane_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/hand_plane"))
hand_plane_marker1 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/hand_plane1"))

# Tracking previous state
previous_accel = None
previous_accel1 = None

# Marker positions (initialize outside the loop)
marker_position = np.array([0.0, 0.0, 0.0])  # Initial position
marker_position1 = np.array([0.0, 0.0, 0.0])  # Initial position

# Marker rotations (initialize outside the loop)
marker_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Initial rotation
marker_rotation1 = np.array([1.0, 0.0, 0.0, 0.0])  # Initial rotation

# Marker translations (initialize outside the loop)
translation = np.array([0.0, 0.0, 0.0])  # Initialize translation
translation1 = np.array([0.0, 0.0, 0.0])  # Initialize translation

# Start the Joy-Con data script as a subprocess
joycon_script_path = "/home/dwijen/Documents/CODE/IsaacLab/wbcd/joycon_reader.py"  # Replace with the actual path
joycon_process = subprocess.Popen(["python", joycon_script_path], stdout=subprocess.PIPE, stderr=sys.stderr, text=True)

# format
'''
                data = {
                    "accel": latest_data['accel'],
                    "gyro": latest_data['gyro'],
                    "accel1": latest_data['accel1'],
                    "gyro1": latest_data['gyro1']
                }
'''

jsonstorage = {
    "accel": None,
    "gyro": None,
    "accel1": None,
    "gyro1": None
}

jsonlock = threading.Lock()

# Function to read data from the subprocess
def read_joycon_data(process):
    global jsonstorage
    
    while True:
        try:
            line = process.stdout.readline().strip()
            if not line:
                break
            try:
                data = json.loads(line)
                with jsonlock:
                    jsonstorage.update(data)
                    print(jsonstorage)
            except json.JSONDecodeError as e:
                print("Failed to parse JSON: ", line)
                print(f"Failed to parse JSON: {e}")
        except Exception as e:
            print(f"Error reading from subprocess: {e}")
            break

# launch thread to read from process
thread = threading.Thread(target=read_joycon_data, args=(joycon_process,))
thread.start()

# Helper function to convert gyro data to quaternion rotation delta
def gyro_to_quaternion(gyro_data, dt=0.01):
    # Scale gyro data (adjust as needed)
    scale = 0.01
    
    # Extract gyro values
    gyro_x = gyro_data[0] * scale
    gyro_y = gyro_data[1] * scale
    gyro_z = gyro_data[2] * scale
    
    # Calculate rotation angle (magnitude)
    angle = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2) * dt
    
    # Handle zero rotation case
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    # Normalize rotation axis
    axis_x = gyro_x / angle
    axis_y = gyro_y / angle
    axis_z = gyro_z / angle
    
    # Create quaternion [w, x, y, z]
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    
    qw = np.cos(half_angle)
    qx = axis_x * sin_half_angle
    qy = axis_y * sin_half_angle
    qz = axis_z * sin_half_angle
    
    return np.array([qw, qx, qy, qz])

# Function to multiply two quaternions
def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


# Main simulation loop
current_pos1 = np.array([0.0, 0.0, 0.0])  # Initial position for first Joy-Con
current_pos2 = np.array([0.0, 1.0, 0.0])  # Initial position for second Joy-Con (offset to distinguish)
current_quat1 = np.array([1.0, 0.0, 0.0, 0.0])  # Initial rotation (identity quaternion)
current_quat2 = np.array([1.0, 0.0, 0.0, 0.0])  # Initial rotation (identity quaternion)

# Constants for signal processing
accel_scale = 0.0001  # Scale for acceleration (adjust as needed)
accel_decay = 0.99    # Decay factor for velocity (simulates friction)
dt = 0.01             # Time step (10ms)

# Velocity vectors
velocity1 = np.array([0.0, 0.0, 0.0])
velocity2 = np.array([0.0, 0.0, 0.0])

for i in range(2000):
    # Read Joy-Con data from the subprocess
    with jsonlock:
        joycon_data = jsonstorage.copy()
    
    if not joycon_data or not any(joycon_data.values()):
        print("No data received from Joy-Con.")
        world.step()
        time.sleep(0.01)  # sleep for 10ms
        continue
    
    # Process first Joy-Con data
    accel = joycon_data.get("accel")
    gyro = joycon_data.get("gyro")
    
    # Process second Joy-Con data
    accel1 = joycon_data.get("accel1")
    gyro1 = joycon_data.get("gyro1")
    
    # Debug print
    if i % 100 == 0:  # Only print every 100 frames to reduce spam
        print("joycon data: ", joycon_data)
    
    # Convert to numpy arrays if data exists
    accel = np.array(accel) if accel else np.array([0.0, 0.0, 0.0])
    gyro = np.array(gyro) if gyro else np.array([0.0, 0.0, 0.0])
    accel1 = np.array(accel1) if accel1 else np.array([0.0, 0.0, 0.0])
    gyro1 = np.array(gyro1) if gyro1 else np.array([0.0, 0.0, 0.0])
    
    # Update rotation for first Joy-Con using gyro data
    if gyro is not None and np.any(gyro):
        delta_quat1 = gyro_to_quaternion(gyro, dt)
        current_quat1 = multiply_quaternions(current_quat1, delta_quat1)
        
        # Normalize quaternion to prevent drift
        current_quat1 = current_quat1 / np.linalg.norm(current_quat1)
    
    # Update rotation for second Joy-Con using gyro data
    if gyro1 is not None and np.any(gyro1):
        delta_quat2 = gyro_to_quaternion(gyro1, dt)
        current_quat2 = multiply_quaternions(current_quat2, delta_quat2)
        
        # Normalize quaternion to prevent drift
        current_quat2 = current_quat2 / np.linalg.norm(current_quat2)
    
    # Update position using accelerometer data for first Joy-Con
    if accel is not None and np.any(accel):
        # Apply rotation to accelerometer data to get world-space acceleration
        # Note: This is a simplified version - full IMU processing would be more complex
        world_accel1 = accel * accel_scale
        
        # Update velocity using acceleration
        velocity1 = velocity1 * accel_decay + world_accel1 * dt
        
        # Update position using velocity
        current_pos1 = current_pos1 + velocity1 * dt
    
    # Update position using accelerometer data for second Joy-Con
    if accel1 is not None and np.any(accel1):
        world_accel2 = accel1 * accel_scale
        velocity2 = velocity2 * accel_decay + world_accel2 * dt
        current_pos2 = current_pos2 + velocity2 * dt
    
    # Convert to tensors for visualization
    marker_position1 = current_pos1
    marker_rotation1 = current_quat1
    
    marker_position2 = current_pos2
    marker_rotation2 = current_quat2
    
    # Create tensors for visualization
    trans_tensor1 = torch.tensor(np.array([marker_position1]))
    quat_tensor1 = torch.tensor(np.array([marker_rotation1]))
    
    trans_tensor2 = torch.tensor(np.array([marker_position2]))
    quat_tensor2 = torch.tensor(np.array([marker_rotation2]))
    
    # Visualize the hand plane markers
    hand_plane_marker.visualize(trans_tensor1, quat_tensor1)
    hand_plane_marker1.visualize(trans_tensor2, quat_tensor2)
    
    # Step the simulation
    world.step()

# Clean up: Terminate the subprocess when the simulation ends
joycon_process.terminate()
simulation_app.close()