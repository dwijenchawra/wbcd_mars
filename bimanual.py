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
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg




logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('hand_tracking')

key_points_lock = threading.Lock()
latest_key_points1 = np.random.rand(21, 3)

def start_process(port):
    command = ['python3', '/home/dwijen/Documents/CODE/IsaacLab/wbcd/mp_depth.py', '--rgb_port', str(port), '--depth_port', str(port+1)]
    print(f"Starting process with command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def read_key_points(process, key_points_var):
    global latest_key_points1
    print(f"Starting to read key points from process {key_points_var}")
    while True:
        line = process.stdout.readline()
        if not line:
            print(f"Process {key_points_var} output none")
            # random key points
            key_points = np.random.rand(21, 3)
            with key_points_lock:
                latest_key_points1 = key_points
            break
        try:
            key_points = json.loads(line.strip())
            with key_points_lock:
                latest_key_points1 = np.array(key_points[0])  # Convert to NumPy array for process 1
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from process {key_points_var}: {line}")
            continue


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

# add seventh joint for gripper
left_joint_positions = np.append(left_joint_positions, 0.0)
right_joint_positions = np.append(right_joint_positions, 0.0)


# Noise parameters
noise_amplitude = 0.05  # Maximum noise in radians (about 3 degrees)

# Create initial ArticulationAction for position control
left_action = ArticulationActions(
    joint_names=None,
    joint_indices=np.arange(7),
    joint_positions=left_joint_positions,
    joint_velocities=None,
    joint_efforts=None
)

right_action = ArticulationActions(
    joint_names=None,
    joint_indices=np.arange(7),
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
handpose_left = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/handpose_left"))

##############
# TODO:
# - list all the joints possible, and see which one is the gripper (right now joint 7 is not working anything)
############

# Main simulation loop
for i in range(1000):
    # Generate random noise for each joint
    left_noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=7)
    right_noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=7)

    # Apply noise to joint positions every 100 frames
    if i % 100 == 0:
        left_joint_positions += left_noise
        right_joint_positions += right_noise

    # Update articulation actions
    left_action.joint_positions = left_joint_positions
    right_action.joint_positions = right_joint_positions

    # Apply actions to articulations
    arm_left.apply_action(left_action)
    arm_right.apply_action(right_action)

    # Rest of the simulation loop (step world, apply actions, etc.)
    world.step()

simulation_app.close()