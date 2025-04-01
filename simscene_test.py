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

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_inv,
    quat_rotate_inverse,
    subtract_frame_transforms,
)


global_stiffness = [15.0, 25.0, 5.0, 0.0, 0.0, 0.0]
global_damping = [10.0, 10.0, 1.0, 0.0, 0.0, 0.0]
velocity_limit = [2.175, 2.175, 2.175, 2.175, 2.175, 2.175]
effort_limit = [400.0, 400.0, 400.0, 400.0, 400.0, 400.0]


@configclass
class BimanualManipulationCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    # world
    world = AssetBaseCfg(
        prim_path="/World/bm_setup",
        spawn=sim_utils.UsdFileCfg(usd_path="/home/dwijen/Documents/CODE/IsaacLab/wbcd/bim_nogripper.usd")
    )

    arm_left = ArticulationCfg(
        prim_path="/World/bm_setup/ur10e",
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
        prim_path="/World/bm_setup/ur10e_01",
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
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                velocity_limit_sim=2.175,
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    

# Update robot states - Simplified to remove force calculation
def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    arm: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
):
    """Update the robot states.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        arm: (Articulation) Robot articulation.
        ee_frame_idx: (int) End-effector frame index.
        arm_joint_ids: (list[int]) Arm joint indices.

    Returns:
        jacobian_b (torch.tensor): Jacobian in the body frame.
        mass_matrix (torch.tensor): Mass matrix.
        gravity (torch.tensor): Gravity vector.
        ee_pose_b (torch.tensor): End-effector pose in the body frame.
        ee_vel_b (torch.tensor): End-effector velocity in the body frame.
        root_pose_w (torch.tensor): Root pose in the world frame.
        ee_pose_w (torch.tensor): End-effector pose in the world frame.
        joint_pos (torch.tensor): The joint positions.
        joint_vel (torch.tensor): The joint velocities.
    """
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = arm.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = arm.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = arm.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(arm.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = arm.data.root_pos_w
    root_quat_w = arm.data.root_quat_w
    ee_pos_w = arm.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = arm.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = arm.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
    root_vel_w = arm.data.root_vel_w  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_rotate_inverse(arm.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_rotate_inverse(arm.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Get joint positions and velocities
    joint_pos = arm.data.joint_pos[:, arm_joint_ids]
    joint_vel = arm.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        joint_pos,
        joint_vel,
    )


# Update the target commands - Simplified to only handle position control
def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    ee_target_set: torch.tensor,
    current_goal_idx: int,
):
    """Update the targets for the operational space controller.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        osc: (OperationalSpaceController) Operational space controller.
        root_pose_w: (torch.tensor) Root pose in the world frame.
        ee_target_set: (torch.tensor) End-effector target set.
        current_goal_idx: (int) Current goal index.

    Returns:
        command (torch.tensor): Updated target command.
        ee_target_pose_b (torch.tensor): Updated target pose in the body frame.
        ee_target_pose_w (torch.tensor): Updated target pose in the world frame.
        next_goal_idx (int): Next goal index.
    """

    # update the ee desired command
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:] = ee_target_set[current_goal_idx]

    # update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b[:] = command[:, :7]  # Only handle pose_abs since that's all we have now

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


# Convert the target commands to the task frame - Simplified for position control only
def convert_to_task_frame(osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor):
    """Converts the target commands to the task frame.

    Args:
        osc: OperationalSpaceController object.
        command: Command to be converted.
        ee_target_pose_b: Target pose in the body frame.

    Returns:
        command (torch.tensor): Target command in the task frame.
        task_frame_pose_b (torch.tensor): Target pose in the task frame.
    """
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    # Handle only pose_abs conversion
    command[:, :3], command[:, 3:7] = subtract_frame_transforms(
        task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
    )

    return command, task_frame_pose_b



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    arm_left: Articulation = scene["arm_left"]
    arm_right: Articulation = scene["arm_right"]
    
    sim_dt = sim.get_physics_dt()
    
    ee_frame_name = "wrist_3_link"
    arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    print(arm_left.joint_names)
    
    # Get frame and joint indices for both arms
    ee_frame_idx_left = arm_left.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids_left = arm_left.find_joints(arm_joint_names)[0]
    
    ee_frame_idx_right = arm_right.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids_right = arm_right.find_joints(arm_joint_names)[0]
    
    # for now, use only one arm
    ee_frame_idx = ee_frame_idx_left
    arm_joint_ids = arm_joint_ids_left
    robot: Articulation = arm_left
    
    # Create the OSC - Modified to use only position control
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],  # Only use position control
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=True,
        motion_damping_ratio_task=1.0,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],  # Control all axes
        nullspace_control="none",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define targets for the arm - Modified to only include position and stiffness
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            [0.5, 0.15, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
            [0.5, -0.3, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
            [0.5, 0.0, 0.5, 0.0, 0.92387953, 0.0, 0.38268343],
        ],
        device=sim.device,
    )
    kp_set_task = torch.tensor(
        [
            [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
            [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
            [320.0, 320.0, 320.0, 320.0, 320.0, 320.0],
        ],
        device=sim.device,
    )
    # Combine pose and stiffness only (no force control)
    ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, kp_set_task], dim=-1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    robot.update(dt=sim_dt)

    # Get the center of the robot soft joint limits
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)
    
    # print joint limits
    print("Joint Limits: ", end="")
    for joint_id in arm_joint_ids:
        joint_limit = robot.data.soft_joint_pos_limits[0, joint_id]
        print(f"{joint_limit[0]:.2f} {joint_limit[1]:.2f}", end=" ")
    print()
    
    # print joint centers
    print("Joint Centers: ", end="")
    for joint_center in joint_centers[0]:
        print(f"{joint_center:.2f}", end=" ")
    print()

    # get the updated states
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        joint_pos,
        joint_vel,
    ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

    # Track the given target command
    current_goal_idx = 0  # Current goal index for the arm
    command = torch.zeros(
        scene.num_envs, osc.action_dim, device=sim.device
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset every 500 steps
        if count % 500 == 0:
            # reset joint state to default
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
            robot.write_data_to_sim()
            robot.reset()
            # reset target pose
            robot.update(sim_dt)
            _, _, _, ee_pose_b, _, _, _, _, _ = update_states(
                sim, scene, robot, ee_frame_idx, arm_joint_ids
            )  # at reset, the jacobians are not updated to the latest state
            command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
                sim, scene, osc, root_pose_w, ee_target_set, current_goal_idx
            )
            # set the osc command
            osc.reset()
            command, task_frame_pose_b = convert_to_task_frame(osc, command=command, ee_target_pose_b=ee_target_pose_b)
            osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)
        else:
            # get the updated states
            (
                jacobian_b,
                mass_matrix,
                gravity,
                ee_pose_b,
                ee_vel_b,
                root_pose_w,
                ee_pose_w,
                joint_pos,
                joint_vel,
            ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)
            
            # print all of the outputs of the previous function on individual lines
            # print("Jacobian (Body Frame):", jacobian_b)
            # print("Mass Matrix:", mass_matrix)
            # print("Gravity Vector:", gravity)
            # print("End-Effector Pose (Body Frame):", ee_pose_b)
            # print("End-Effector Velocity (Body Frame):", ee_vel_b)
            # print("Root Pose (World Frame):", root_pose_w)
            # print("End-Effector Pose (World Frame):", ee_pose_w)
            # print("Joint Positions:", joint_pos)
            # print("Joint Velocities:", joint_vel)
            
            # Set zero force for the force feedback in compute since we're only doing position control
            ee_force_b = torch.zeros(scene.num_envs, 3, device=sim.device)
            
            # compute the joint commands
            joint_efforts = osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                current_ee_force_b=ee_force_b,  # Pass zero force
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=joint_centers,
            )
            # scale down efforts by 0.1
            joint_efforts *= 0.1

            print("Joint Efforts: ", end="")
            for effort in joint_efforts[0]:
                print(f"{effort:.2f}", end=" ")
            print()
            
            
                        
            # apply actions
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()

        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

        # perform step
        sim.step(render=True)
        # update robot buffers
        robot.update(sim_dt)
        # update buffers
        scene.update(sim_dt)
        # update sim-time
        count += 1


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
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()