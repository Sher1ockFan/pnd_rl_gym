from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from pndbotics_sdk_py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from pndbotics_sdk_py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__LowCmd_, pnd_adam_msg_dds__LowState_, pnd_adam_msg_dds__HandCmd_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowCmd_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowState_ 
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import HandCmd_

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_adam
from common.rotation_helper import get_gravity_orientation, transform_imu_data, ypr_to_quaternion
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # All 23 motor indices in SDK order
        self.all_motor_idx = config.leg_joint2motor_idx + config.arm_waist_joint2motor_idx
        self.all_kps = np.array(config.kps, dtype=np.float32)
        self.all_kds = np.array(config.kds, dtype=np.float32)
        self.all_default = config.default_angles.copy()

        num_actions = config.num_actions
        history_length = config.history_length
        self.joint_ids_map = config.joint_ids_map

        # Per-term observation history buffers (same layout as deploy_mujoco_23dof)
        self.ang_vel_history = np.zeros((history_length, 3), dtype=np.float32)
        self.gravity_history = np.zeros((history_length, 3), dtype=np.float32)
        self.cmd_history = np.zeros((history_length, 3), dtype=np.float32)
        self.jpos_history = np.zeros((history_length, num_actions), dtype=np.float32)
        self.jvel_history = np.zeros((history_length, num_actions), dtype=np.float32)
        self.action_history = np.zeros((history_length, num_actions), dtype=np.float32)

        self.action = np.zeros(num_actions, dtype=np.float32)
        self.target_dof_pos = self.all_default.copy()
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.counter = 0

        # Load policy
        self.policy = torch.jit.load(config.policy_path, map_location="cpu")
        self.policy.eval()

        # SDK / DDS setup
        if config.msg_type == "adam_lite":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(23)
            self.low_state = pnd_adam_msg_dds__LowState_(23)
        elif config.msg_type == "adam_sp":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(29)
            self.low_state = pnd_adam_msg_dds__LowState_(29)
            self.hand_cmd = pnd_adam_msg_dds__HandCmd_()
            self.close_hand = np.array([500] * 12, dtype=int)
            self.hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
            self.hand_pub.Init()
        else:
            raise ValueError(f"Invalid msg_type: {config.msg_type}")

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self.LowState_Handler, 10)

        self.wait_for_low_state()

        if config.msg_type == "adam_lite":
            init_cmd_adam(self.low_cmd)

        print(f"[deploy_real] num_obs={config.num_obs}  num_actions={num_actions}  "
              f"history={history_length}  control_dt={config.control_dt}")

    # ---- SDK callbacks ----

    def LowState_Handler(self, msg: LowState_):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmd_):
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick != 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # ---- state machine ----

    def zero_torque_state(self):
        print("Enter zero torque state.  Waiting for [Start]...")
        while self.remote_controller.button[KeyMap.start] != 1:
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.all_motor_idx
        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for step in range(num_step):
            alpha = step / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + self.all_default[j] * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.all_kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self.all_kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            if self.config.msg_type != "adam_lite":
                for i in range(12):
                    self.hand_cmd.position[i] = self.close_hand[i]
                self.hand_pub.Write(self.hand_cmd)

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.  Waiting for [A]...")
        dof_idx = self.all_motor_idx
        while self.remote_controller.button[KeyMap.A] != 1:
            for j in range(len(dof_idx)):
                motor_idx = dof_idx[j]
                self.low_cmd.motor_cmd[motor_idx].q = self.all_default[j]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.all_kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self.all_kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            if self.config.msg_type != "adam_lite":
                for i in range(12):
                    self.hand_cmd.position[i] = self.close_hand[i]
                self.hand_pub.Write(self.hand_cmd)

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # ---- main control loop step ----

    def run(self):
        if self.config.msg_type != "adam_lite":
            for i in range(12):
                self.hand_cmd.position[i] = self.close_hand[i]
            self.hand_pub.Write(self.hand_cmd)

        self.counter += 1
        cfg = self.config

        # --- Read joint states (SDK order, 23 joints) ---
        qj = np.zeros(cfg.num_actions, dtype=np.float32)
        dqj = np.zeros(cfg.num_actions, dtype=np.float32)
        for i, motor_idx in enumerate(self.all_motor_idx):
            qj[i] = self.low_state.motor_state[motor_idx].q
            dqj[i] = self.low_state.motor_state[motor_idx].dq

        # --- Read IMU ---
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        if cfg.imu_type == "torso":
            waist_motor = cfg.arm_waist_joint2motor_idx[0]
            waist_yaw = self.low_state.motor_state[waist_motor].q
            waist_yaw_omega = self.low_state.motor_state[waist_motor].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat, imu_omega=ang_vel.reshape(1, 3),
            )
            ang_vel = ang_vel.flatten()

        # --- Remote command ---
        self.cmd[0] = self.remote_controller.get_walk_x_direction_speed()
        self.cmd[1] = self.remote_controller.get_walk_y_direction_speed()
        self.cmd[2] = self.remote_controller.get_walk_yaw_direction_speed()

        # --- Build scaled observation terms ---
        cur_ang_vel = ang_vel * cfg.ang_vel_scale
        cur_gravity = get_gravity_orientation(quat)
        cur_cmd = self.cmd * cfg.cmd_scale
        # Reorder SDK→policy order via joint_ids_map
        cur_jpos = ((qj - self.all_default) * cfg.dof_pos_scale)[self.joint_ids_map]
        cur_jvel = (dqj * cfg.dof_vel_scale)[self.joint_ids_map]

        # --- Shift history: drop oldest, append newest ---
        self.ang_vel_history[:-1] = self.ang_vel_history[1:]
        self.ang_vel_history[-1] = cur_ang_vel
        self.gravity_history[:-1] = self.gravity_history[1:]
        self.gravity_history[-1] = cur_gravity
        self.cmd_history[:-1] = self.cmd_history[1:]
        self.cmd_history[-1] = cur_cmd
        self.jpos_history[:-1] = self.jpos_history[1:]
        self.jpos_history[-1] = cur_jpos
        self.jvel_history[:-1] = self.jvel_history[1:]
        self.jvel_history[-1] = cur_jvel
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = self.action

        # --- Concatenate observation (same order as training) ---
        obs = np.concatenate([
            self.ang_vel_history.flatten(),
            self.gravity_history.flatten(),
            self.cmd_history.flatten(),
            self.jpos_history.flatten(),
            self.jvel_history.flatten(),
            self.action_history.flatten(),
        ])

        # --- Policy inference ---
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
            self.action = self.policy(obs_tensor).squeeze().numpy()

        # --- Apply action: reorder policy→SDK, then compute target ---
        self.target_dof_pos = self.all_default.copy()
        self.target_dof_pos[self.joint_ids_map] += self.action * cfg.action_scale

        # --- Send motor commands (SDK order) ---
        for j, motor_idx in enumerate(self.all_motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = self.target_dof_pos[j]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.all_kps[j]
            self.low_cmd.motor_cmd[motor_idx].kd = self.all_kds[j]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(cfg.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder",
                        default="adam_lite_23dof.yaml")
    args = parser.parse_args()

    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(1, args.net)

    controller = Controller(config)

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.B] == 1:
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
