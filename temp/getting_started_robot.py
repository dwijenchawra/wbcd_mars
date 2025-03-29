# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys

import carb
import numpy as np
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path


# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()
    
assets_root_path = "omniverse://localhost/NVIDIA/Assets/Isaac/4.5"
print("Assets root path: ", assets_root_path)

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

# Add Franka
asset_path = "wbcd/ur5.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")  # add robot to stage
arm: Articulation = Articulation(prim_paths_expr="/World/Arm", name="my_arm")  # create an articulation object

add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm2")  # add robot to stage
arm2: Articulation = Articulation(prim_paths_expr="/World/Arm2", name="my_arm2")  # create an articulation object


# set the initial poses of the arm and the car so they don't collide BEFORE the simulation starts
arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())

arm2.set_world_poses(positions=np.array([[0.0, -1.0, 0.0]]) / get_stage_units())

# GoPro Hero9 Max Lens Mod Configuration (2.7K mode)
width, height = 2704, 1520  # 2.7K resolution (16:9 aspect ratio)

# Camera matrix (fx, fy) in pixels, (cx, cy) optical center
camera_matrix = [
    [943.8, 0.0, 1352.0],  # fx=943.8, cx=2704/2
    [0.0, 942.5, 760.0],   # fy=942.5, cy=1520/2
    [0.0, 0.0, 1.0]
]

# Distortion coefficients (k1, k2, p1, p2) - estimated for Max Lens Mod
distortion_coefficients = [0.32, -0.28, 0.003, 0.001]

# Sensor characteristics
pixel_size = 2.33  # microns (6.3mm sensor width / 2704 pixels)
f_stop = 2.8       # GoPro's typical aperture in low light
focus_distance = 1.2  # meters (action sports hyperfocal distance)
diagonal_fov = 155   # degrees (Max Lens Mod spec)


camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.0, 0.0, 2.0]),  # 1 meter away from the side of the cube
    frequency=30,
    resolution=(width, height),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
)
# Initialize camera BEFORE setting properties
camera.initialize()

simulation_app.update()  # Required for USD stage initialization

# Calculate the focal length and aperture size from the camera matrix
((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix
horizontal_aperture = pixel_size * 1e-3 * width
vertical_aperture = pixel_size * 1e-3 * height
focal_length_x = fx * pixel_size * 1e-3
focal_length_y = fy * pixel_size * 1e-3
focal_length = (focal_length_x + focal_length_y) / 2  # in mm

# Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
camera.set_focal_length(focal_length / 10.0)
camera.set_focus_distance(focus_distance)
camera.set_lens_aperture(f_stop * 100.0)
camera.set_horizontal_aperture(horizontal_aperture / 10.0)
camera.set_vertical_aperture(vertical_aperture / 10.0)

camera.set_clipping_range(0.05, 1.0e5)

# Set the distortion coefficients
camera.set_projection_type("fisheyePolynomial")
camera.set_kannala_brandt_properties(width, height, cx, cy, diagonal_fov, distortion_coefficients)


# Finalize
camera.initialize()

# initialize the world
my_world.reset()

num_joints = 6
num_envs = 1

for i in range(5000):
    # step the simulation, both rendering and physics
    my_world.step(render=True)
    if i % 50 == 0:
        positions = np.random.uniform(low=-3.0, high=3.0, size=(num_envs, num_joints))
        arm.set_joint_positions(positions)
        arm2.set_joint_positions(positions)
    

simulation_app.close()
