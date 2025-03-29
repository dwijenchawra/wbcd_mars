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
translation = np.array([0.0, 0.0, 0.0])  # Initialize translation
translation1 = np.array([0.0, 0.0, 0.0])  # Initialize translation

# Start the Joy-Con data script as a subprocess
joycon_script_path = "/home/dwijen/Documents/CODE/IsaacLab/wbcd/joycon_reader.py"  # Replace with the actual path
joycon_process = subprocess.Popen(["python", joycon_script_path], stdout=subprocess.PIPE, stderr=sys.stderr, text=True)

jsonstorage = {
    "accel": None,
    "gyro": None,
    "buttons": None,
    "analog-sticks": None,
    "battery": None,
    "accel1": None,
    "gyro1": None,
    "buttons1": None,
    "analog-sticks1": None,
    "battery1": None
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
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
        except Exception as e:
            print(f"Error reading from subprocess: {e}")
            break

# launch thread to read from process
thread = threading.Thread(target=read_joycon_data, args=(joycon_process,))
thread.start()

# Initialize variables to track previous sensor data
previous_accel = None
previous_accel1 = None
previous_gyro = None
previous_gyro1 = None

# Initialize marker positions and orientations
marker_position = np.array([0.0, 0.0, 0.0])
marker_position1 = np.array([0.0, 0.0, 0.0])
marker_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion [w, x, y, z]
marker_rotation1 = np.array([1.0, 0.0, 0.0, 0.0])

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
    accel = None
    gyro = None
    
    # Extract accel data for first Joy-Con
    accel_data = joycon_data.get("accel")
    if accel_data:
        accel_x = accel_data.get("x", 0)
        accel_y = accel_data.get("y", 0)
        accel_z = accel_data.get("z", 0)
        accel = np.array([accel_x, accel_y, accel_z])
    else:
        print("Accel data not found for first Joy-Con.")
    
    # Extract gyro data for first Joy-Con
    gyro_data = joycon_data.get("gyro")
    if gyro_data:
        gyro_x = gyro_data.get("x", 0)
        gyro_y = gyro_data.get("y", 0)
        gyro_z = gyro_data.get("z", 0)
        gyro = np.array([gyro_x, gyro_y, gyro_z])
    else:
        print("Gyro data not found for first Joy-Con.")
    
    # Process second Joy-Con data
    accel1 = None
    gyro1 = None
    
    # Extract accel data for second Joy-Con
    accel_data1 = joycon_data.get("accel1")
    if accel_data1:
        accel_x1 = accel_data1.get("x", 0)
        accel_y1 = accel_data1.get("y", 0)
        accel_z1 = accel_data1.get("z", 0)
        accel1 = np.array([accel_x1, accel_y1, accel_z1])
    else:
        print("Accel data not found for second Joy-Con.")
    
    # Extract gyro data for second Joy-Con
    gyro_data1 = joycon_data.get("gyro1")
    if gyro_data1:
        gyro_x1 = gyro_data1.get("x", 0)
        gyro_y1 = gyro_data1.get("y", 0)
        gyro_z1 = gyro_data1.get("z", 0)
        gyro1 = np.array([gyro_x1, gyro_y1, gyro_z1])
    else:
        print("Gyro data not found for second Joy-Con.")
    
    # Update first marker position based on accelerometer data
    if previous_accel is not None and accel is not None:
        # Calculate acceleration delta
        accel_delta = accel - previous_accel
        
        # Scale the delta to control marker movement speed
        translation_delta = accel_delta * 0.0001  # Adjust the scaling factor as needed
        marker_position += translation_delta
    
    # Update first marker rotation based on gyroscope data
    if gyro is not None:
        # Convert gyro data to quaternion rotation
        rotation_delta = gyro_to_quaternion(gyro)
        
        # Apply rotation delta to current rotation
        marker_rotation = multiply_quaternions(marker_rotation, rotation_delta)
        
        # Normalize quaternion to prevent drift
        marker_rotation = marker_rotation / np.linalg.norm(marker_rotation)
    
    # Update second marker position based on accelerometer data
    if previous_accel1 is not None and accel1 is not None:
        # Calculate acceleration delta
        accel_delta1 = accel1 - previous_accel1
        
        # Scale the delta to control marker movement speed
        translation_delta1 = accel_delta1 * 0.0001  # Adjust the scaling factor as needed
        marker_position1 += translation_delta1
    
    # Update second marker rotation based on gyroscope data
    if gyro1 is not None:
        # Convert gyro data to quaternion rotation
        rotation_delta1 = gyro_to_quaternion(gyro1)
        
        # Apply rotation delta to current rotation
        marker_rotation1 = multiply_quaternions(marker_rotation1, rotation_delta1)
        
        # Normalize quaternion to prevent drift
        marker_rotation1 = marker_rotation1 / np.linalg.norm(marker_rotation1)
    
    # Visualize the first marker
    if accel is not None:
        # Create tensors for transformation
        trans_tensor = torch.tensor(np.array([marker_position]))
        quat_tensor = torch.tensor(np.array([marker_rotation]))
        
        # Visualize the hand plane marker
        hand_plane_marker.visualize(trans_tensor, quat_tensor)
    
    # Visualize the second marker
    if accel1 is not None:
        # Create tensors for transformation
        trans_tensor1 = torch.tensor(np.array([marker_position1]))
        quat_tensor1 = torch.tensor(np.array([marker_rotation1]))
        
        # Visualize the hand plane marker
        hand_plane_marker1.visualize(trans_tensor1, quat_tensor1)
    
    # Store current sensor data for the next iteration
    previous_accel = accel
    previous_accel1 = accel1
    previous_gyro = gyro
    previous_gyro1 = gyro1
    
    # Optional: Print transform information
    if i % 50 == 0:  # Print every 50 frames to avoid spam
        print(f"Frame {i}:")
        print("Marker 1 - Position:", marker_position)
        print("Marker 1 - Rotation:", marker_rotation)
        if accel1 is not None:
            print("Marker 2 - Position:", marker_position1)
            print("Marker 2 - Rotation:", marker_rotation1)
    
    # Step the simulation
    world.step()


# Clean up: Terminate the subprocess when the simulation ends
joycon_process.terminate()
simulation_app.close()