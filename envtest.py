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
from isaaclab.sensors import CameraCfg, TiledCamera, TiledCameraCfg

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
        spawn=sim_utils.UsdFileCfg(usd_path="/home/dwijen/Documents/CODE/IsaacLab/wbcd/envtest.usd"),
    )

    arm_left = ArticulationCfg(
        prim_path="/World/bm_setup/left/left_arm",
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
            "finger_joint": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                velocity_limit_sim=None,
                effort_limit_sim=None,
                stiffness=0,
                damping=0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    arm_right = ArticulationCfg(
        prim_path="/World/bm_setup/right/right_arm",
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
            "finger_joint": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                velocity_limit_sim=None,
                effort_limit_sim=None,
                stiffness=0,
                damping=0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


# Helper Functions
def initialize_controllers_and_markers(scene: InteractiveScene, sim: sim_utils.SimulationContext):
    """Initializes Differential IK controllers and visualization markers."""
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.1}
    )
    diff_ik_controller_left = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    diff_ik_controller_right = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker_left = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current_left"))
    goal_marker_left = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_left"))
    ee_marker_right = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current_right"))
    goal_marker_right = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_right"))

    markers = {
        "ee_left": ee_marker_left,
        "goal_left": goal_marker_left,
        "ee_right": ee_marker_right,
        "goal_right": goal_marker_right,
    }
    controllers = {"left": diff_ik_controller_left, "right": diff_ik_controller_right}
    return controllers, markers


def setup_robot_entities(scene: InteractiveScene):
    """Sets up and resolves SceneEntityCfg for both robots."""
    robot_entity_cfg_left = SceneEntityCfg(
        "arm_left",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        body_names=["tool0"],
    )
    robot_entity_cfg_right = SceneEntityCfg(
        "arm_right",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        body_names=["tool0"],
    )
    robot_entity_cfg_left.resolve(scene)
    robot_entity_cfg_right.resolve(scene)

    # Calculate Jacobian indices (adjusting for fixed base)
    robot_left: Articulation = scene["arm_left"]
    robot_right: Articulation = scene["arm_right"]
    ee_jacobi_idx_left = robot_entity_cfg_left.body_ids[0] - 1 if robot_left.is_fixed_base else robot_entity_cfg_left.body_ids[0]
    ee_jacobi_idx_right = robot_entity_cfg_right.body_ids[0] - 1 if robot_right.is_fixed_base else robot_entity_cfg_right.body_ids[0]
    
    # Set up cameras
    camera_left = TiledCamera(
        TiledCameraCfg(
            prim_path="/World/bm_setup/left/left_arm/ur10e/tool0/camera",
            width=1280,
            height=720,
            spawn=None
        )
    )
    camera_right = TiledCamera(
        TiledCameraCfg(
            prim_path="/World/bm_setup/right/right_arm/ur10e/tool0/camera",
            width=1280,
            height=720,
            spawn=None
        )
    )
    cameras = {"left": camera_left, "right": camera_right}            
    entities = {"left": robot_entity_cfg_left, "right": robot_entity_cfg_right}
    jacobi_indices = {"left": ee_jacobi_idx_left, "right": ee_jacobi_idx_right}
    return entities, jacobi_indices, cameras


def reset_simulation_state(
    scene: InteractiveScene,
    robot_left: Articulation,
    robot_right: Articulation,
    controllers: dict,
    entities: dict,
    ik_commands_left: torch.Tensor,
    ik_commands_right: torch.Tensor,
):
    """Resets the robots, controllers, and commands to their initial states."""
    # Reset robots
    default_joint_pos_left = robot_left.data.default_joint_pos.clone()
    default_joint_vel_left = robot_left.data.default_joint_vel.clone()
    robot_left.write_joint_state_to_sim(default_joint_pos_left, default_joint_vel_left)
    robot_left.reset()

    default_joint_pos_right = robot_right.data.default_joint_pos.clone()
    default_joint_vel_right = robot_right.data.default_joint_vel.clone()
    robot_right.write_joint_state_to_sim(default_joint_pos_right, default_joint_vel_right)
    robot_right.reset()

    # Reset commands and controllers
    ik_commands_left[:] = 0.0
    ik_commands_right[:] = 0.0
    controllers["left"].reset()
    controllers["right"].reset()

    # Store initial desired joint positions
    joint_pos_des = {
        "left": default_joint_pos_left[:, entities["left"].joint_ids].clone(),
        "right": default_joint_pos_right[:, entities["right"].joint_ids].clone(),
    }
    return joint_pos_des


def get_joycon_commands(joycons: JoyconInterface, device: str):
    """Gets pose commands from Joycons and converts them to tensors."""
    left_pos_np, left_rot_np, right_pos_np, right_rot_np = joycons.get_lr_pos_rot_safe()

    # TODO: Replace placeholder logic with actual Joycon -> Target Pose mapping
    # Using fixed target for demonstration
    left_pos_np = torch.tensor([0.5, 0.0, 0.5], device=device).unsqueeze(0)  # Add batch dim
    left_rot_np = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0) # Add batch dim, ensure valid quat
    right_pos_np = torch.tensor([0.5, 0.0, 0.5], device=device).unsqueeze(0) # Add batch dim
    right_rot_np = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0) # Add batch dim, ensure valid quat

    # Example using actual Joycon data (needs scaling/mapping)
    # scale_pos = 0.1
    # left_pos = torch.tensor(left_pos_np * scale_pos, device=device, dtype=torch.float).unsqueeze(0)
    # left_rot = torch.tensor(left_rot_np, device=device, dtype=torch.float).unsqueeze(0) # Assuming joycon gives w,x,y,z
    # right_pos = torch.tensor(right_pos_np * scale_pos, device=device, dtype=torch.float).unsqueeze(0)
    # right_rot = torch.tensor(right_rot_np, device=device, dtype=torch.float).unsqueeze(0) # Assuming joycon gives w,x,y,z
    
    left_pos = torch.tensor(left_pos_np, device=device)
    left_rot = torch.tensor(left_rot_np, device=device)
    right_pos = torch.tensor(right_pos_np, device=device)
    right_rot = torch.tensor(right_rot_np, device=device)


    return left_pos, left_rot, right_pos, right_rot


def get_simulation_data(robot: Articulation, entity_cfg: SceneEntityCfg, ee_jacobi_idx: int):
    """Extracts relevant data (Jacobian, poses, joint positions) from the simulation for a single robot."""
    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, entity_cfg.joint_ids]
    ee_pose_w = robot.data.body_state_w[:, entity_cfg.body_ids[0], 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]
    joint_pos = robot.data.joint_pos[:, entity_cfg.joint_ids]
    return jacobian, ee_pose_w, root_pose_w, joint_pos


def compute_ik_and_apply_actions(
    controller: DifferentialIKController,
    robot: Articulation,
    entity_cfg: SceneEntityCfg,
    jacobian: torch.Tensor,
    ee_pose_w: torch.Tensor,
    root_pose_w: torch.Tensor,
    current_joint_pos: torch.Tensor,
    ik_command_pos: torch.Tensor,
    ik_command_rot: torch.Tensor,
):
    """Computes IK, sets target, and returns desired joint positions and world target pose."""
    # Compute EE pose in base frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    # Compute target EE pose in world frame
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_command_pos, ik_command_rot
    )

    # Compute IK
    joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

    # Apply action
    robot.set_joint_position_target(joint_pos_des, joint_ids=entity_cfg.joint_ids)

    return joint_pos_des, ee_target_pos_w, ee_target_quat_w


def update_visualizations(
    ee_marker: VisualizationMarkers,
    goal_marker: VisualizationMarkers,
    robot: Articulation,
    entity_cfg: SceneEntityCfg,
    ee_target_pos_w: torch.Tensor | None,
    ee_target_quat_w: torch.Tensor | None,
    ik_command_pos: torch.Tensor,
    ik_command_rot: torch.Tensor,
    scene: InteractiveScene,
):
    """Updates the visualization markers for EE and goal poses."""
    # Get current EE pose
    ee_pose_w = robot.data.body_state_w[:, entity_cfg.body_ids[0], 0:7]
    ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

    # Visualize goal
    if ee_target_pos_w is not None and ee_target_quat_w is not None:
        goal_marker.visualize(ee_target_pos_w, ee_target_quat_w)
    else:
        # Fallback for the first frame or if target wasn't computed in world frame yet
        goal_marker.visualize(ik_command_pos + scene.env_origins, ik_command_rot)

# Main Simulation Function
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot_left: Articulation = scene["arm_left"]
    robot_right: Articulation = scene["arm_right"]

    # Initialization
    controllers, markers = initialize_controllers_and_markers(scene, sim)
    entities, jacobi_indices = setup_robot_entities(scene)
    joycons = JoyconInterface()

    # Create buffers
    ik_commands_left = torch.zeros(scene.num_envs, controllers["left"].action_dim, device=sim.device)
    ik_commands_right = torch.zeros(scene.num_envs, controllers["right"].action_dim, device=sim.device)
    joint_pos_des = {"left": None, "right": None} # Store desired joint positions
    ee_target_pose_w = {"left": (None, None), "right": (None, None)} # Store target poses (pos, quat)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    print("[INFO]: Simulation setup complete. Starting run loop...")
    # Simulation loop
    while simulation_app.is_running():
        # Reset on first step
        if count == 0:
            joint_pos_des = reset_simulation_state(
                scene, robot_left, robot_right, controllers, entities, ik_commands_left, ik_commands_right
            )

        # -- Get input commands
        left_cmd_pos, left_cmd_rot, right_cmd_pos, right_cmd_rot = get_joycon_commands(joycons, sim.device)

        # -- Set commands for controllers
        ik_commands_left[:, 0:3] = left_cmd_pos
        ik_commands_left[:, 3:7] = left_cmd_rot
        ik_commands_right[:, 0:3] = right_cmd_pos
        ik_commands_right[:, 3:7] = right_cmd_rot
        controllers["left"].set_command(ik_commands_left)
        controllers["right"].set_command(ik_commands_right)

        # -- Get current simulation data
        jacobian_left, ee_pose_w_left, root_pose_w_left, current_joint_pos_left = get_simulation_data(
            robot_left, entities["left"], jacobi_indices["left"]
        )
        jacobian_right, ee_pose_w_right, root_pose_w_right, current_joint_pos_right = get_simulation_data(
            robot_right, entities["right"], jacobi_indices["right"]
        )

        # -- Compute IK and apply actions
        joint_pos_des["left"], ee_target_pos_w_left, ee_target_quat_w_left = compute_ik_and_apply_actions(
            controllers["left"], robot_left, entities["left"], jacobian_left, ee_pose_w_left, root_pose_w_left,
            current_joint_pos_left, ik_commands_left[:, 0:3], ik_commands_left[:, 3:7]
        )
        joint_pos_des["right"], ee_target_pos_w_right, ee_target_quat_w_right = compute_ik_and_apply_actions(
            controllers["right"], robot_right, entities["right"], jacobian_right, ee_pose_w_right, root_pose_w_right,
            current_joint_pos_right, ik_commands_right[:, 0:3], ik_commands_right[:, 3:7]
        )
        ee_target_pose_w["left"] = (ee_target_pos_w_left, ee_target_quat_w_left)
        ee_target_pose_w["right"] = (ee_target_pos_w_right, ee_target_quat_w_right)

        # -- Write data and step simulation
        scene.write_data_to_sim()
        sim.step()

        # -- Update scene buffers and visualization
        scene.update(sim_dt)
        update_visualizations(
            markers["ee_left"], markers["goal_left"], robot_left, entities["left"],
            ee_target_pose_w["left"][0], ee_target_pose_w["left"][1],
            ik_commands_left[:, 0:3], ik_commands_left[:, 3:7], scene
        )
        update_visualizations(
            markers["ee_right"], markers["goal_right"], robot_right, entities["right"],
            ee_target_pose_w["right"][0], ee_target_pose_w["right"][1],
            ik_commands_right[:, 0:3], ik_commands_right[:, 3:7], scene
        )
        
        # get camera images
        # image_left = cameras["left"].data.output

        # Increment counter
        count += 1


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
    print("[INFO]: Starting main simulation...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    print("[INFO]: Closing simulation app...")
    simulation_app.close()
    print("[INFO]: Simulation closed.")
