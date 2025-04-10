# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="ur10", help="Name of the robot.")
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

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

from joycon_interface import JoyconInterface

# global_stiffness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# global_damping = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# global_stiffness = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# global_damping = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

global_stiffness = [800.0, 800.0, 800.0, 800.0, 800.0, 800.0]
global_damping = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
velocity_limit = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
effort_limit = [87.0, 87.0, 87.0, 87.0, 87.0, 87.0]

@configclass
class BimanualManipulationCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    # world
    world = AssetBaseCfg(
        prim_path="/World/bm_setup",
        spawn=sim_utils.UsdFileCfg(usd_path="/home/dwijen/Documents/CODE/IsaacLab/wbcd/bim_nogripper.usd"),
    )

    arm_left = ArticulationCfg(
        prim_path="/World/bm_setup/ur10_instanceable",
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
            "shoulder_pan_joint": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint"],
                velocity_limit_sim=velocity_limit[0],
                effort_limit_sim=effort_limit[0],
                stiffness=global_stiffness[0],
                damping=global_damping[0],
            ),
            "shoulder_lift_joint": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_lift_joint"],
                velocity_limit_sim=velocity_limit[1],
                effort_limit_sim=effort_limit[1],
                stiffness=global_stiffness[1],
                damping=global_damping[1],
            ),
            "elbow_joint": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                velocity_limit_sim=velocity_limit[2],
                effort_limit_sim=effort_limit[2],
                stiffness=global_stiffness[2],
                damping=global_damping[2],
            ),
            "wrist_1_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_1_joint"],
                velocity_limit_sim=velocity_limit[3],
                effort_limit_sim=effort_limit[3],
                stiffness=global_stiffness[3],
                damping=global_damping[3],
            ),
            "wrist_2_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_2_joint"],
                velocity_limit_sim=velocity_limit[4],
                effort_limit_sim=effort_limit[4],
                stiffness=global_stiffness[4],
                damping=global_damping[4],
            ),
            "wrist_3_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_3_joint"],
                velocity_limit_sim=velocity_limit[5],
                effort_limit_sim=effort_limit[5],
                stiffness=global_stiffness[5],
                damping=global_damping[5],
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    arm_right = ArticulationCfg(
        prim_path="/World/bm_setup/ur10_instanceable_01",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                # "finger_joint": 0.0,
            },
        ),
        actuators={
            "shoulder_pan_joint": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint"],
                velocity_limit_sim=velocity_limit[0],
                effort_limit_sim=effort_limit[0],
                stiffness=global_stiffness[0],
                damping=global_damping[0],
            ),
            "shoulder_lift_joint": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_lift_joint"],
                velocity_limit_sim=velocity_limit[1],
                effort_limit_sim=effort_limit[1],
                stiffness=global_stiffness[1],
                damping=global_damping[1],
            ),
            "elbow_joint": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                velocity_limit_sim=velocity_limit[2],
                effort_limit_sim=effort_limit[2],
                stiffness=global_stiffness[2],
                damping=global_damping[2],
            ),
            "wrist_1_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_1_joint"],
                velocity_limit_sim=velocity_limit[3],
                effort_limit_sim=effort_limit[3],
                stiffness=global_stiffness[3],
                damping=global_damping[3],
            ),
            "wrist_2_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_2_joint"],
                velocity_limit_sim=velocity_limit[4],
                effort_limit_sim=effort_limit[4],
                stiffness=global_stiffness[4],
                damping=global_damping[4],
            ),
            "wrist_3_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_3_joint"],
                velocity_limit_sim=velocity_limit[5],
                effort_limit_sim=effort_limit[5],
                stiffness=global_stiffness[5],
                damping=global_damping[5],
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    # robot = scene["robot"]
    robot_left: Articulation = scene["arm_left"]
    robot_right: Articulation = scene["arm_right"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.1})
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    diff_ik_controller_right = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    
    # Markers for right arm
    ee_marker_right = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current_right"))
    goal_marker_right = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_right"))
    
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot_left.device)
    ik_commands_right = torch.zeros(scene.num_envs, diff_ik_controller_right.action_dim, device=robot_right.device)
        
    robot_entity_cfg = SceneEntityCfg(
        "arm_left",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], 
        body_names=["ee_link"])
    
    robot_entity_cfg_right = SceneEntityCfg(
        "arm_right",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], 
        body_names=["ee_link"])
    
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    robot_entity_cfg_right.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot_left.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]            
    if robot_right.is_fixed_base:
        ee_jacobi_idx_right = robot_entity_cfg_right.body_ids[0] - 1
    else:
        ee_jacobi_idx_right = robot_entity_cfg_right.body_ids[0]       

    joycons = JoyconInterface() # start the joycon reader

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    print("[INFO]: Simulation started...")
    # Simulation loop
    ee_target_pos_w, ee_target_quat_w = None, None
    while simulation_app.is_running():

        if count == 0:
            # reset joint state
            joint_pos = robot_left.data.default_joint_pos.clone()
            joint_vel = robot_left.data.default_joint_vel.clone()
            robot_left.write_joint_state_to_sim(joint_pos, joint_vel)
            robot_left.reset()
            
            joint_pos_right = robot_right.data.default_joint_pos.clone()
            joint_vel_right = robot_right.data.default_joint_vel.clone()
            robot_right.write_joint_state_to_sim(joint_pos_right, joint_vel_right)
            robot_right.reset()
            
            # reset actions
            ik_commands[:] = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=sim.device)
            ik_commands_right[:] = torch.zeros(scene.num_envs, diff_ik_controller_right.action_dim, device=sim.device)
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            joint_pos_des_right = joint_pos_right[:, robot_entity_cfg_right.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller_right.reset()
            # diff_ik_controller.set_command(ik_commands)
            ###############
        
        # set get IK command from joycon values and convert to tensors on sim.device
        left_pos, left_rot, right_pos, right_rot = joycons.get_lr_pos_rot_safe()
        left_pos = torch.tensor(left_pos, device=sim.device)
        left_rot = torch.tensor(left_rot, device=sim.device)
        right_pos = torch.tensor(right_pos, device=sim.device)
        right_rot = torch.tensor(right_rot, device=sim.device)
                
                
        #ignore pos for now, just add 1 to z
        left_pos = torch.tensor([0.5, 0.0, 0.5], device=sim.device)
        right_pos = torch.tensor([0.5, 0.0, 0.5], device=sim.device)
        # set the ik command
        ik_commands[:, 0:3] = left_pos
        ik_commands[:, 3:7] = left_rot
        ik_commands_right[:, 0:3] = right_pos
        ik_commands_right[:, 3:7] = right_rot
        
        diff_ik_controller.set_command(ik_commands)
        diff_ik_controller_right.set_command(ik_commands_right)
            
        # obtain quantities from simulation
        jacobian = robot_left.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        jacobian_right = robot_right.root_physx_view.get_jacobians()[:, ee_jacobi_idx_right, :, robot_entity_cfg_right.joint_ids]
        
        ee_pose_w = robot_left.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_pose_w_right = robot_right.data.body_state_w[:, robot_entity_cfg_right.body_ids[0], 0:7]
        root_pose_w = robot_left.data.root_state_w[:, 0:7]
        root_pose_w_right = robot_right.data.root_state_w[:, 0:7]
        joint_pos = robot_left.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_pos_right = robot_right.data.joint_pos[:, robot_entity_cfg_right.joint_ids]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pos_b_right, ee_quat_b_right = subtract_frame_transforms(
            root_pose_w_right[:, 0:3], root_pose_w_right[:, 3:7], ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7]
        )
        
        # compute ee goal in world frame
        ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_commands[:, 0:3], ik_commands[:, 3:7]
        )
        ee_target_pos_w_right, ee_target_quat_w_right = combine_frame_transforms(
            root_pose_w_right[:, 0:3], root_pose_w_right[:, 3:7], ik_commands_right[:, 0:3], ik_commands_right[:, 3:7]
        )
        
        # compute the joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        joint_pos_des_right = diff_ik_controller_right.compute(ee_pos_b_right, ee_quat_b_right, jacobian_right, joint_pos_right)

        robot_left.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot_right.set_joint_position_target(joint_pos_des_right, joint_ids=robot_entity_cfg_right.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot_left.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_pose_w_right = robot_right.data.body_state_w[:, robot_entity_cfg_right.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        ee_marker_right.visualize(ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7])

        if ee_target_pos_w is not None:
            goal_marker.visualize(ee_target_pos_w, ee_target_quat_w)
            goal_marker_right.visualize(ee_target_pos_w_right, ee_target_quat_w_right)
        else:
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
            goal_marker_right.visualize(ik_commands_right[:, 0:3] + scene.env_origins, ik_commands_right[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([-2.5, 0.4, 3.5], [0.0, 0.4, 1.0])
    # Design scene
    scene_cfg = BimanualManipulationCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
