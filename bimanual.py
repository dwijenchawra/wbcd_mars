#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
import numpy as np
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions

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

# Main simulation loop
for i in range(1000):
    # Generate random noise for each joint
    left_noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=6)
    right_noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=6)

    # Update actions with noisy positions
    left_action.joint_positions = left_joint_positions + left_noise
    right_action.joint_positions = right_joint_positions + right_noise

    # Apply control actions
    arm_left.apply_action(left_action)
    arm_right.apply_action(right_action)

    # Step simulation
    world.step()

simulation_app.close()
