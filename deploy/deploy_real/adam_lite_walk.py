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
from common.basic_func import LowPassFilter
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.device = torch.device("cpu")

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path, map_location="cpu")
        
        # Initialize estimator model if available
        self.est_model = None
        if config.estimatory_path:
            self.est_model = torch.jit.load(config.estimatory_path, map_location="cpu")
        
        # Observation dimensions
        self.num_obs = config.num_obs
        self.input_num = config.num_obs * config.frame_stack + config.latent_size
        self.est_input_num = config.num_obs * config.latent_frame_stack
        self.delta_num = 0
        
        # Initialize observation buffers
        self.hist_obs = np.zeros(self.est_input_num, dtype=np.float32)
        self.vae_obs = np.zeros(self.est_input_num, dtype=np.float32)
        # Low pass filters
        self.omega_filter = LowPassFilter(100, 0.707, config.control_dt, 3)
        self.action_filter = LowPassFilter(100, 0.707, config.control_dt, 23)

        self.timer_plan = 0.0

        self.action_last = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        # Action smoothing parameters
        self.last_action_d = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.last_action_dot_d = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.predictive_time = 0.02
        self.para_0 = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.para_1 = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.para_2 = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.para_3 = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)

        self.input_data_mlp_humanoid = np.zeros(self.num_obs, dtype=np.float32)
        self.output_data_mlp = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.est_input_array = np.zeros(self.est_input_num, dtype=np.float32)
        self.input_array = np.zeros(self.input_num, dtype=np.float32)
        
        # Initializing process variables
        self.cur_joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.cur_joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        self.gait_d = "Walk"  # "Stand" or "Walk"
        self.gait_a = "Walk"

        # Command and state variables
        self.cmd = np.array([0.0, 0.0, 0.0])
        
        # Command scales
        self.command_scales_humanoid = np.array([2.0, 2.0, 2.0])
        
        # Gait and phase variables
        self.rl_counter = 0
        self.phase_counter = 1  # Increment by 1 each inference step

        self.gait_cycle_humanoid = config.cycle_time_walk
        self.avg_yaw_vel = 0.0

        self.input_data_est_tensor = torch.zeros((1, self.est_input_array.size), dtype=torch.float32, device=self.device)
        self.input_data_tensor = torch.zeros((1, self.input_array.size), dtype=torch.float32, device=self.device)
        self.output_data = torch.zeros(self.config.num_actions)

        # Observation scales (from C++ code)
        self.obs_scales_dof_pos = config.dof_pos_scale
        self.obs_scales_dof_vel = config.dof_vel_scale
        self.obs_scales_ang_vel = config.ang_vel_scale
        self.action_scales = config.action_scale
        
        self.counter = 0
        self.control_dt_ns = int(self.config.control_dt * 1e9)
        
        self.next_tick_ns = None  # first call will init
        if config.msg_type == "adam_lite":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(23)
            self.low_state = pnd_adam_msg_dds__LowState_(23)
        elif config.msg_type == "adam_pro":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(31)
            self.low_state = pnd_adam_msg_dds__LowState_(31)
            self.hand_cmd = pnd_adam_msg_dds__HandCmd_()
            self.close_hand = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int)
            self.hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
            self.hand_pub.Init()
        else:
            raise ValueError("Invalid msg_type")
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmd_)
        self.lowcmd_publisher_.Init()


        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self.LowState_Handler, 10)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "adam_lite":
            init_cmd_adam(self.low_cmd)
            # init_cmd_adam(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def LowState_Handler(self, msg: LowState_):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmd_):
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick != 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            # create_zero_cmd(self.low_cmd)
            # self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):

        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx

        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            # hand publisher
            if config.msg_type != "adam_lite":
                for i in range(12):
                    self.hand_cmd.position[i] = self.close_hand[i]
                self.hand_pub.Write(self.hand_cmd)

            self.send_cmd(self.low_cmd)
            self.last_send_time = time.time()

            time.sleep(self.config.control_dt)
    



    def default_pos_state(self):
            
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # hand publisher
            if config.msg_type != "adam_lite":
                for i in range(12):
                    self.hand_cmd.position[i] = self.close_hand[i]
                self.hand_pub.Write(self.hand_cmd)
            self.compute_obervation()
            self.compute_action()
            # print("wu:")
            # create observation
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def compute_obervation(self):
        self.obs_start_time = time.time()
        """Compute observations similar to C++ StateMLP::computeObs"""
        # Get joint positions and velocities
                # Fill leg joints
        for i in range(self.config.num_actions):
            self.cur_joint_pos[i] = self.low_state.motor_state[i].q
            self.cur_joint_vel[i] = self.low_state.motor_state[i].dq

        # Inference period: 100Hz inference from 400Hz control (every 4 control cycles)
        
        # Compute gait phase
        gait_phase = self.rl_counter * 0.01 / self.gait_cycle_humanoid
        self.sin_phase = np.sin(2.0 * np.pi * gait_phase)
        self.cos_phase = np.cos(2.0 * np.pi * gait_phase)
        self.ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        quat = ypr_to_quaternion(self.low_state.imu_state.ypr[0],self.low_state.imu_state.ypr[1],self.low_state.imu_state.ypr[2])
        if self.config.imu_type == "torso":
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, self.ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=self.ang_vel)
      
        # Update average yaw velocity
        self.avg_yaw_vel = (1.0 * 0.01 / self.gait_cycle_humanoid) * self.ang_vel[0][2] + \
                          (1.0 - 1.0 * 0.01 / self.gait_cycle_humanoid) * self.avg_yaw_vel
        quat = self.low_state.imu_state.quaternion
        # create observation
        self.gravity_orientation = get_gravity_orientation(quat)
        # Construct observation vector
        self.input_data_mlp_humanoid[0] = self.cmd[0] * self.command_scales_humanoid[0]
        self.input_data_mlp_humanoid[1] = self.cmd[1] * self.command_scales_humanoid[1]
        self.input_data_mlp_humanoid[2] = self.cmd[2] * self.command_scales_humanoid[2]
        self.input_data_mlp_humanoid[3:6] = self.gravity_orientation
        self.input_data_mlp_humanoid[6:6+self.config.num_actions] = (self.cur_joint_pos - self.config.default_angles) * self.obs_scales_dof_pos
        self.input_data_mlp_humanoid[6+self.config.num_actions:6+self.config.num_actions*2] = self.cur_joint_vel * self.obs_scales_dof_vel
        self.input_data_mlp_humanoid[6+self.config.num_actions*2:6+self.config.num_actions*3] = self.action_last
        self.input_data_mlp_humanoid[6+self.config.num_actions*3:9+self.config.num_actions*3] = self.ang_vel * self.obs_scales_ang_vel
        self.input_data_mlp_humanoid[9+self.config.num_actions*3] = self.avg_yaw_vel * self.obs_scales_ang_vel
        self.input_data_mlp_humanoid[10+self.config.num_actions*3] = self.sin_phase
        self.input_data_mlp_humanoid[11+self.config.num_actions*3] = self.cos_phase
        

        # Apply low pass filter to angular velocity
        # omega_segment = self.input_data_mlp_humanoid[6+self.config.num_actions*3:6+self.config.num_actions*3+3]
        # filtered_omega = self.omega_filter.update(omega_segment)
        # self.input_data_mlp_humanoid[6+self.config.num_actions*3:6+self.config.num_actions*3+3] = filtered_omega
        
        # Clip observation
        self.input_data_mlp_humanoid = np.clip(self.input_data_mlp_humanoid, -18.0, 18.0)
        # Update sliding window for vae_obs
        num_obs = self.num_obs
        self.vae_obs[:-num_obs] = self.vae_obs[num_obs:]
        self.vae_obs[-num_obs:] = self.input_data_mlp_humanoid[:num_obs]
        self.est_input_array[:] = self.vae_obs
        
        # Update sliding window for hist_obs
        self.hist_obs[:-num_obs] = self.hist_obs[num_obs:]
        self.hist_obs[-num_obs:] = self.input_data_mlp_humanoid[:num_obs]
        self.input_array[:self.est_input_num] = self.hist_obs
        self.obs_end_time = time.time()

    def compute_action(self):
        self.action_start_time = time.time()
        # Run estimator model
        # --- compute_action() 中 ---
        if self.est_model is not None:
            # 直接更新 tensor 的值，而不是每次新建
            self.cp_est_time_start = time.time()
            self.input_data_est_tensor[0].copy_(torch.from_numpy(self.est_input_array).float())
            self.cp_est_time_end = time.time()

            self.est_infer_time_start = time.time()
            with torch.no_grad():
                output_data_est = self.est_model(self.input_data_est_tensor)
            self.est_infer_time_end = time.time()

            # Append latent to input_array
        self.input_array[self.est_input_num : self.est_input_num + self.config.latent_size] = output_data_est[0, :self.config.latent_size].cpu().numpy()

        # --- policy 推理 ---
        self.cp_policy_time_start = time.time()

        self.input_data_tensor[0].copy_(torch.from_numpy(self.input_array).float())
        self.cp_policy_time_end = time.time()

        self.policy_infer_time_start = time.time()
        with torch.no_grad():
            self.output_data = (
                self.policy(self.input_data_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
            )

            self.policy_infer_time_end = time.time()

        # out = np.array([output_data[0][i].item() for i in range(output_data.size(1))])
        # # Process output
        # kObsDof = self.config.num_actions
        # for i in range(kObsDof):
        # #     # Compute action smoothing parameters for Walk gait (after processing all joints)
        # #     if self.gait_a == "Walk":
        #     self.output_data_mlp[i] = np.clip(out[i], -18.0, 18.0)
        # #         self.para_0 = self.last_action_d
        # #         self.para_1 = self.last_action_dot_d
        # #         self.para_2 = 3.0 * (self.output_data_mlp - self.last_action_d - self.last_action_dot_d * self.predictive_time) / \
        # #                     self.predictive_time / self.predictive_time - \
        # #                     (-self.last_action_dot_d) / self.predictive_time

        # #         self.para_3 = -2.0 * (self.output_data_mlp - self.last_action_d -
        # #                             self.last_action_dot_d * self.predictive_time) / \
        # #                     self.predictive_time / self.predictive_time / self.predictive_time + \
        # #                     (-self.last_action_dot_d) / self.predictive_time / self.predictive_time
        # #         self.timer_plan = 0.0

        self.rl_counter += self.phase_counter
        self.action_end_time = time.time()
    def run(self):
        # ---------- init tick anchor ----------
        run_start = time.time()

        # ======================
        # your original logic
        # ======================
        if config.msg_type != "adam_lite":
            for i in range(12):
                self.hand_cmd.position[i] = self.close_hand[i]
            self.hand_pub.Write(self.hand_cmd)

        self.cmd[0] = self.remote_controller.get_walk_x_direction_speed()
        self.cmd[1] = self.remote_controller.get_walk_y_direction_speed()
        self.cmd[2] = self.remote_controller.get_walk_yaw_direction_speed()

        # if self.counter % 4 == 0:
        self.compute_obervation()
        self.compute_action()

        self.action_last = self.output_data
        # self.timer_plan += self.config.control_dt
        # t = self.timer_plan

        # mlp_out = (
        #     self.para_0
        #     + self.para_1 * t
        #     + self.para_2 * t * t
        #     + self.para_3 * t * t * t
        # )

        # mlp_out_dot = (
        #     self.para_1
        #     + 2.0 * self.para_2 * t
        #     + 3.0 * self.para_3 * t * t
        # )

        self.mlp_out_scaled = self.output_data * self.action_scales + self.config.default_angles


        # self.mlp_out_scaled[5]  = self.low_state.motor_state[5].q
        # self.mlp_out_scaled[11] = self.low_state.motor_state[11].q

        for i in range(self.config.num_actions):
            self.low_cmd.motor_cmd[i].q = float(self.mlp_out_scaled[i])
            self.low_cmd.motor_cmd[i].qd = 0.0
            self.low_cmd.motor_cmd[i].kp = float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = float(self.config.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.send_cmd(self.low_cmd)

        # self.last_action_d = self.output_data_mlp
        # self.last_action_dot_d = mlp_out_dot
        self.counter += 1

        # ======================
        # precise timing control (align to absolute schedule)
        # ======================
        time_until_next_step = self.config.control_dt - (time.time() - run_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        else:
            print("cp_est_time", (self.cp_est_time_end - self.cp_est_time_start))
            print("cp_policy_time", (self.cp_policy_time_end - self.cp_policy_time_start))
            print("est_infer_time", (self.est_infer_time_end - self.est_infer_time_start))
            print("policy_infer_time", (self.policy_infer_time_end - self.policy_infer_time_start))
            # print("action_time", (self.action_end_time - self.action_start_time))
            # print(time.time() - run_start)
        # else:
        #     # overrun: reset anchor to avoid accumulating delay
        #     self.next_tick_ns = now_ns + self.control_dt_ns

        # optional debug


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="adam_lite.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(1, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.B] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    # create_damping_cmd(controller.low_cmd)
    # controller.send_cmd(controller.low_cmd)
    print("Exit")
