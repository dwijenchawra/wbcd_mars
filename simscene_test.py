# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse
from pty import spawn

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.universal_robots import UR10_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaacsim.core.prims import Articulation
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.utils.articulations import find_all_articulation_base_paths
from pxr import UsdPhysics





@configclass
class BimanualManipulationCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    #world
    world = AssetBaseCfg(
        # prim_path="/World/envs/env_.*/bm_setup",
        prim_path="/World/bm_setup",
        spawn=sim_utils.UsdFileCfg(usd_path="/home/dwijen/Documents/CODE/IsaacLab/wbcd/bimanualworld.usd")
    )
    
    
    arm_left = ArticulationCfg(
        prim_path="/World/bm_setup/ur10e_robotiq2f_140",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit_sim=100.0,
                effort_limit_sim=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

    arm_right = ArticulationCfg(
        prim_path="/World/bm_setup/ur10e_robotiq2f_141",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit_sim=100.0,
                effort_limit_sim=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )    


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    arm_left = scene["arm_left"]
    arm_right = scene["arm_right"]
        
    # # list all articulation roots under world 
    # prim_paths = prims_utils.find_matching_prim_paths("/World/bm_setup/.*robotiq.*/")
    # print(prim_paths)
    
    # print("All articulation base paths:")
    # print(find_all_articulation_base_paths())
    
    # for prim_path in prim_paths:
    #     prim = prims_utils.get_prim_at_path(prim_path)
    #     UsdPhysics.ArticulationRootAPI.Apply(prim)

    # # ['/World/bm_setup/ur10e_robotiq2f_140', '/World/bm_setup/ur10e_robotiq2f_141']

    # arm_left_prim_path = "/World/bm_setup/ur10e_robotiq2f_140"
    # arm_right_prim_path = "/World/bm_setup/ur10e_robotiq2f_141"

    # # arm_left_cfg = ArticulationCfg(
    # #     prim_path=arm_left_prim_path,
    # #     init_state=ArticulationCfg.InitialStateCfg(
    # #         joint_pos={
    # #             "shoulder_pan_joint": 0.0,
    # #             "shoulder_lift_joint": -1.712,
    # #             "elbow_joint": 1.712,
    # #             "wrist_1_joint": 0.0,
    # #             "wrist_2_joint": 0.0,
    # #             "wrist_3_joint": 0.0,
    # #             "finger_joint": 0.0,
    # #         },
    # #     ),
    # #     actuators={
    # #         "arm": ImplicitActuatorCfg(
    # #             joint_names_expr=[".*"],
    # #             velocity_limit_sim=100.0,
    # #             effort_limit_sim=87.0,
    # #             stiffness=800.0,
    # #             damping=40.0,
    # #         ),
    # #     },
    # # )

    # # arm_right_cfg = ArticulationCfg(
    # #     prim_path=arm_right_prim_path,
    # #     init_state=ArticulationCfg.InitialStateCfg(
    # #         joint_pos={
    # #             "shoulder_pan_joint": 0.0,
    # #             "shoulder_lift_joint": -1.712,
    # #             "elbow_joint": 1.712,
    # #             "wrist_1_joint": 0.0,
    # #             "wrist_2_joint": 0.0,
    # #             "wrist_3_joint": 0.0,
    # #             "finger_joint": 0.0,
    # #         },
    # #     ),
    # #     actuators={
    # #         "arm": ImplicitActuatorCfg(
    # #             joint_names_expr=[".*"],
    # #             velocity_limit_sim=100.0,
    # #             effort_limit_sim=87.0,
    # #             stiffness=800.0,
    # #             damping=40.0,
    # #         ),
    # #     },
    # # )

    # arm_left: Articulation = Articulation(arm_left_prim_path)
    # arm_right: Articulation = Articulation(arm_right_prim_path)
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        # if count % 500 == 0:
        #     # reset counter
        #     count = 0
        #     # reset the scene entities
        #     # root state
        #     # we offset the root state by the origin since the states are written in simulation world frame
        #     # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
        #     root_state = robot.data.default_root_state.clone()
        #     root_state[:, :3] += scene.env_origins
        #     robot.write_root_pose_to_sim(root_state[:, :7])
        #     robot.write_root_velocity_to_sim(root_state[:, 7:])
        #     # set joint positions with some noise
        #     joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        #     joint_pos += torch.rand_like(joint_pos) * 0.1
        #     robot.write_joint_state_to_sim(joint_pos, joint_vel)
        #     # clear internal buffers
        #     scene.reset()
        #     print("[INFO]: Resetting robot state...")
        # # Apply random action
        # -- generate random joint efforts
        efforts_l = torch.randn(arm_left.num_joints) * 5.0
        efforts_r = torch.randn(arm_right.num_joints) * 5.0
        arm_left.set_joint_effort_target(efforts_l)
        arm_right.set_joint_effort_target(efforts_r)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    physics_context = sim.get_physics_context()
    
    # disable gpu dynamics
    physics_context.enable_gpu_dynamics(False)
    
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = BimanualManipulationCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    time.sleep(10)
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()