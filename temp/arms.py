# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



##
# Pre-defined configs
##
# isort: off
from isaaclab_assets import (
    FRANKA_PANDA_CFG,
    UR10_CFG,
    KINOVA_JACO2_N7S300_CFG,
    KINOVA_JACO2_N6S300_CFG,
    KINOVA_GEN3_N7_CFG,
    SAWYER_CFG,
)

ur10_with_gripper = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

gripper = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UR10/ur10e_robotiq2f-140.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     joint_pos={
    #         "shoulder_pan_joint": 0.0,
    #         "shoulder_lift_joint": -1.712,
    #         "elbow_joint": 1.712,
    #         "wrist_1_joint": 0.0,
    #         "wrist_2_joint": 0.0,
    #         "wrist_3_joint": 0.0,
    #     },
    # ),
    # actuators={
    #     "arm": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         velocity_limit=100.0,
    #         effort_limit=87.0,
    #         stiffness=800.0,
    #         damping=40.0,
    #     ),
    # },
)

ur10_with_gripper = UR10_CFG

# isort: on


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()

def create_camera(path):
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
        prim_path=path,
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



def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=6, spacing=3.0)
    
    
    
    
    


    # Origin 1 with original ur10
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func("/World/Origin1/Table", cfg, translation=(0.00, 0.0, 1.05)) # x was 0.55
    # -- Robot
    franka_arm_cfg = ur10_with_gripper.replace(prim_path="/World/Origin1/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    
    gripper_one = gripper.replace(prim_path="/World/Origin1/Robot/gripper")
    gripperfinal = Articulation(cfg=gripper_one)

    # robot_assembler = RobotAssembler()
    # assembled_robot = robot_assembler.assemble_articulations(base_robot_path="/World/Origin1/Robot",
    #                                                     attach_robot_path="/World/Origin1/Robot/gripper",
    #                                                     base_robot_mount_frame="/link_ee",
    #                                                     attach_robot_mount_frame="/link_m",
    #                                                     mask_all_collisions = True,
    #                                                     single_robot=False)

    franka_panda = Articulation(cfg=franka_arm_cfg)    
    

    # Origin 2 with UR10
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func("/World/Origin2/Table", cfg, translation=(0.0, 0.0, 1.03))
    # -- Robot
    ur10_cfg = ur10_with_gripper.replace(prim_path="/World/Origin2/Robot")
    ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    ur10 = Articulation(cfg=ur10_cfg)



    

    # return the scene information
    scene_entities = {
        "franka_panda": franka_panda,
        "ur10": ur10
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # set joint positions
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
            joint_pos_target = joint_pos_target.clamp_(
                robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
            )
            # apply action to the robot
            robot.set_joint_position_target(joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)
            
        # camera.get_current_frame()
        # img = Image.fromarray(camera.get_rgba()[:, :, :3])



def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    create_camera()
    
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
