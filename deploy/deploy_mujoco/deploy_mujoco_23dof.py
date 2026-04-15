import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        history_length = config["history_length"]

        # MuJoCo actuator/joint indices for the policy-controlled joints
        # (full XML may have extra joints like wrist yaw that the policy does not control)
        mj_joint_indices = np.array(config["mj_joint_indices"], dtype=np.int32)

        # Policy↔SDK joint reordering: joint_ids_map[i] = SDK index for policy output i
        # If not provided, assume policy order == SDK/MuJoCo order (identity)
        _ids_map = config.get("joint_ids_map", None)
        if _ids_map is not None:
            joint_ids_map = np.array(_ids_map, dtype=np.int32)
        else:
            joint_ids_map = np.arange(num_actions, dtype=np.int32)

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # qpos/qvel index arrays for extracting policy joint states from MuJoCo
    qpos_indices = mj_joint_indices + 7   # skip floating-base pos(3) + quat(4)
    qvel_indices = mj_joint_indices + 6   # skip floating-base lin_vel(3) + ang_vel(3)

    # Per-term observation history buffers
    # Isaac Lab flattens each term's history independently, then concatenates all terms.
    # Layout: [term1_h0, term1_h1, ..., term1_hN, term2_h0, ..., termK_hN]
    ang_vel_history = np.zeros((history_length, 3), dtype=np.float32)
    gravity_history = np.zeros((history_length, 3), dtype=np.float32)
    cmd_history = np.zeros((history_length, 3), dtype=np.float32)
    jpos_history = np.zeros((history_length, num_actions), dtype=np.float32)
    jvel_history = np.zeros((history_length, num_actions), dtype=np.float32)
    action_history = np.zeros((history_length, num_actions), dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Initialize joint positions to default standing pose (matches training reset)
    d.qpos[qpos_indices] = default_angles
    mujoco.mj_forward(m, d)

    # Load policy
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    print(f"[deploy] num_obs={num_obs}  num_actions={num_actions}  "
          f"history={history_length}  dt={simulation_dt}  decimation={control_decimation}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            tau = pd_control(
                target_dof_pos,
                d.qpos[qpos_indices],
                kps,
                np.zeros_like(kds),
                d.qvel[qvel_indices],
                kds,
            )
            d.ctrl[mj_joint_indices] = tau

            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[qpos_indices]
                dqj = d.qvel[qvel_indices]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                cur_ang_vel = omega * ang_vel_scale
                cur_gravity = get_gravity_orientation(quat)
                cur_cmd = cmd * cmd_scale
                cur_jpos = ((qj - default_angles) * dof_pos_scale)[joint_ids_map]
                cur_jvel = (dqj * dof_vel_scale)[joint_ids_map]

                # Shift history: drop oldest frame, append newest
                ang_vel_history[:-1] = ang_vel_history[1:]
                ang_vel_history[-1] = cur_ang_vel
                gravity_history[:-1] = gravity_history[1:]
                gravity_history[-1] = cur_gravity
                cmd_history[:-1] = cmd_history[1:]
                cmd_history[-1] = cur_cmd
                jpos_history[:-1] = jpos_history[1:]
                jpos_history[-1] = cur_jpos
                jvel_history[:-1] = jvel_history[1:]
                jvel_history[-1] = cur_jvel
                action_history[:-1] = action_history[1:]
                action_history[-1] = action

                obs = np.concatenate([
                    ang_vel_history.flatten(),
                    gravity_history.flatten(),
                    cmd_history.flatten(),
                    jpos_history.flatten(),
                    jvel_history.flatten(),
                    action_history.flatten(),
                ])

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                    action = policy(obs_tensor).squeeze().numpy()

                target_dof_pos = default_angles.copy()
                target_dof_pos[joint_ids_map] += action * action_scale

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
