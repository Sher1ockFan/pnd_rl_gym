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

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_adam, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data, ypr_to_quaternion
from common.remote_controller import RemoteController, KeyMap
from common.basic_func import LowPassFilter
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

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
        self.delta_num = config.delta_num if config.delta_num else 0
        
        # Initialize observation buffers
        self.obs = np.zeros(self.input_num, dtype=np.float32)
        self.vae_obs = np.zeros(self.est_input_num, dtype=np.float32)
        self.hist_obs = np.zeros(self.est_input_num, dtype=np.float32)
        self.est_input_array = np.zeros(self.est_input_num, dtype=np.float32)
        self.input_array = np.zeros(self.input_num, dtype=np.float32)
        self.input_data_mlp_humanoid = np.zeros(self.num_obs, dtype=np.float32)
        
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.action_last = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.output_data_mlp = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        
        # Command and state variables
        self.cmd = np.array([0.0, 0.0, 0.0])
        self.joystick_command = np.array([0.0, 0.0, 0.0])
        self.command_ori = np.array([0.0, 0.0, 0.0])
        self.x_vel_command_offset = 0.0
        self.y_vel_command_offset = 0.0
        self.pos_percent = 0.0
        self.pos_duration = 1.0
        
        # Command scales
        self.command_scales_humanoid = np.array([2.0, 2.0, 2.0])
        
        # Gait and phase variables
        self.rl_counter = 0
        self.stand_counter = 100
        self.phase_counter = 1
        self.gait_d = "Stand"  # "Stand" or "Walk"
        self.gait_a = "Stand"
        self.gait_cycle_humanoid = config.cycle_time_walk if config.cycle_time_walk else 1.0
        self.avg_yaw_vel = 0.0
        
        # Low pass filters
        self.omega_filter = LowPassFilter(100, 0.707, 0.0025, 3)
        self.action_filter = LowPassFilter(100, 0.707, 0.0025, 23)
        
        # Action smoothing parameters
        self.last_action_d = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.last_action_dot_d = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.predictive_time = 0.02
        self.timer_plan = 0.0
        self.para_0 = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.para_1 = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.para_2 = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        self.para_3 = np.zeros(config.num_actions + self.delta_num, dtype=np.float32)
        
        # Observation scales (from C++ code)
        self.obs_scales_dof_pos = config.dof_pos_scale
        self.obs_scales_dof_vel = config.dof_vel_scale
        self.obs_scales_ang_vel = config.ang_vel_scale
        self.action_scales = config.action_scale
        
        self.counter = 0
       
        self.swing_in_air =  np.array([-0.5, 0, -0.25, 0.75, -0.35, 0,
                                        0.5, 0, -0.25, 0.75, -0.35, 0,
                                       0, 0, 0,
                                       0.3, 0, 0, -1.4, 0, 
                                      -0.3, 0, 0, -1.4, 0
                                     ])

        self.default_angles = np.array([0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 
                                        0.0,  0.0,  0.0,  0.0, 0.0, 0.0])
        
        self.arm_waist_target = np.array([0.0, 0.0, 0.0, #waist
                                          0.0, 0.0, 0.0,-1.4,0.0, #left arm
                                          0.0, 0.0, 0.0,-1.4,0.0])
        if config.msg_type == "adam_lite":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(25)
            self.low_state = pnd_adam_msg_dds__LowState_(25)
        elif config.msg_type == "adam_pro":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(31)
            self.low_state = pnd_adam_msg_dds__LowState_(31)
            self.hand_cmd = pnd_adam_msg_dds__HandCmd_()
            self.close_hand = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int)
            self.hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
            self.hand_pub.Init()
        else:
            raise ValueError("Invalid msg_type")
        self.mode_pr_ = MotorMode.PR
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
        # print("Received low state.")
        # self.mode_machine_ = self.low_state.mode_machine
        # print("wireless_remote raw:", self.low_state.wireless_remote)
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmd_):
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        # while self.low_state.tick == 0:
        time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):

        if self.config.msg_type != "adam_lite":
            for i in range(12):
                self.hand_cmd.position[i] = self.close_hand[i]
            self.hand_pub.Write(self.hand_cmd)

        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx

        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.default_angles, self.arm_waist_target), axis=0)
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
            self.send_cmd(self.low_cmd)

            time.sleep(self.config.control_dt)

    def default_pos_state(self):

        if self.config.msg_type != "adam_lite":
            for i in range(12):
                self.hand_cmd.position[i] = self.close_hand[i]
            self.hand_pub.Write(self.hand_cmd)
            
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        
        # Initialize by running computeObs and computeActions a few times
        for i in range(5):
            self.computeObs()
            self.computeActions()
        self.gait_d = "Stand"
        self.rl_counter = self.stand_counter

    def computeObs(self):
        """Compute observations similar to C++ StateMLP::computeObs"""
        # Get joint positions and velocities
        # Fill leg joints
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
        
        # Fill arm/waist joints if needed (for full 23 DOF observation)
        if len(self.config.leg_joint2motor_idx) < self.config.num_actions:
            arm_waist_start_idx = len(self.config.leg_joint2motor_idx)
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                if arm_waist_start_idx + i < self.config.num_actions:
                    self.qj[arm_waist_start_idx + i] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].q
                    self.dqj[arm_waist_start_idx + i] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].dq

        # Get IMU data
        quat = ypr_to_quaternion(
            self.low_state.imu_state.ypr[0],
            self.low_state.imu_state.ypr[1],
            self.low_state.imu_state.ypr[2]
        )
        quat = np.array(quat)  # (w, x, y, z)
        ang_vel_raw = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        
        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel_raw = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel_raw
            )
            ang_vel_raw = ang_vel_raw.reshape(1, -1)[0]

        # Compute rotation matrix and gravity orientation
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        Rb_w = rot.as_matrix()
        gravity_orientation = -Rb_w[:, 2]  # Negative of z-axis
        
        # Transform angular velocity
        rpy = np.array([self.low_state.imu_state.ypr[2], 
                       self.low_state.imu_state.ypr[1], 
                       self.low_state.imu_state.ypr[0]])  # roll, pitch, yaw
        R_xyz_omega = np.eye(3)
        R_xyz_omega[1, :] = R.from_euler('x', rpy[0]).as_matrix()[1, :]
        R_xyz_omega[2, :] = (R.from_euler('x', rpy[0]) * R.from_euler('y', rpy[1])).as_matrix()[2, :]
        ang_vel = Rb_w.T @ R_xyz_omega @ ang_vel_raw
        
        # Update average yaw velocity
        self.avg_yaw_vel = (1.0 * 0.01 / self.gait_cycle_humanoid) * ang_vel[2] + \
                          (1.0 - 1.0 * 0.01 / self.gait_cycle_humanoid) * self.avg_yaw_vel
        
        # Compute gait phase
        gait_phase = self.rl_counter * 0.01 / self.gait_cycle_humanoid
        sin_phase = np.sin(2.0 * np.pi * gait_phase)
        cos_phase = np.cos(2.0 * np.pi * gait_phase)
        
        # Build input_data_mlp_humanoid (single frame observation)
        cur_joint_pos = self.qj.copy()
        cur_joint_vel = self.dqj.copy()
        def_dof_pos = self.config.default_angles
        
        # Construct observation vector
        self.input_data_mlp_humanoid[0] = self.cmd[0] * self.command_scales_humanoid[0]
        self.input_data_mlp_humanoid[1] = self.cmd[1] * self.command_scales_humanoid[1]
        self.input_data_mlp_humanoid[2] = self.cmd[2] * self.command_scales_humanoid[2]
        self.input_data_mlp_humanoid[3:6] = gravity_orientation
        self.input_data_mlp_humanoid[6:6+len(cur_joint_pos)] = (cur_joint_pos - def_dof_pos) * self.obs_scales_dof_pos
        self.input_data_mlp_humanoid[6+len(cur_joint_pos):6+len(cur_joint_pos)*2] = cur_joint_vel * self.obs_scales_dof_vel
        self.input_data_mlp_humanoid[6+len(cur_joint_pos)*2:6+len(cur_joint_pos)*3] = self.action_last[:len(cur_joint_pos)]
        self.input_data_mlp_humanoid[6+len(cur_joint_pos)*3:6+len(cur_joint_pos)*3+3] = ang_vel * self.obs_scales_ang_vel
        self.input_data_mlp_humanoid[6+len(cur_joint_pos)*3+3] = self.avg_yaw_vel * self.obs_scales_ang_vel
        self.input_data_mlp_humanoid[6+len(cur_joint_pos)*3+4] = sin_phase
        self.input_data_mlp_humanoid[6+len(cur_joint_pos)*3+5] = cos_phase
        
        # Apply low pass filter to angular velocity
        omega_segment = self.input_data_mlp_humanoid[6+len(cur_joint_pos)*3:6+len(cur_joint_pos)*3+3]
        filtered_omega = self.omega_filter.update(omega_segment)
        self.input_data_mlp_humanoid[6+len(cur_joint_pos)*3:6+len(cur_joint_pos)*3+3] = filtered_omega
        
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

    def computeActions(self):
        """Compute actions using estimator and policy network (based on C++ StateMLP::computeActions)"""
        # Run estimator model
        if self.est_model is not None:
            input_data_est = torch.from_numpy(self.est_input_array).float().unsqueeze(0)
            with torch.no_grad():
                output_data_est = self.est_model(input_data_est).toTensor()

            # Append latent to input_array
            for i in range(self.config.latent_size):
                self.input_array[self.est_input_num + i] = output_data_est[0][i].item()

        # Run policy network
        input_data = torch.from_numpy(self.input_array).float().unsqueeze(0)
        with torch.no_grad():
            output_data = self.policy(input_data).toTensor()
            out = np.array([output_data[0][i].item() for i in range(output_data.size(1))])

        # Process output
        kObsDof = len(self.config.leg_joint2motor_idx)
        for i in range(kObsDof):
            if self.gait_a == "Walk":
                self.output_data_mlp[i] = np.clip(out[i], -18.0, 18.0)

        # Compute action smoothing parameters for Walk gait (after processing all joints)
        if self.gait_a == "Walk":
            self.para_0 = self.last_action_d.copy()
            self.para_1 = self.last_action_dot_d.copy()
            self.para_2 = 3.0 * (self.output_data_mlp - self.last_action_d -
                                 self.last_action_dot_d * self.predictive_time) / \
                         (self.predictive_time * self.predictive_time) - \
                         (-self.last_action_dot_d) / self.predictive_time
            self.para_3 = -2.0 * (self.output_data_mlp - self.last_action_d -
                                  self.last_action_dot_d * self.predictive_time) / \
                          (self.predictive_time * self.predictive_time * self.predictive_time) + \
                          (-self.last_action_dot_d) / (self.predictive_time * self.predictive_time)
            self.timer_plan = 0.0

        self.rl_counter += self.phase_counter

        # Compute scaled output (similar to C++ mlp_out_scaled)
        self.mlp_out_scaled = self.output_data_mlp * self.action_scales + self.config.default_angles

    def changeState(self):
        """Change gait state based on command norm"""
        norm_cmd = np.sqrt(self.cmd[0]**2 + self.cmd[1]**2 + self.cmd[2]**2)
        
        # Compute current velocity for state transition
        from scipy.spatial.transform import Rotation as R
        quat = ypr_to_quaternion(
            self.low_state.imu_state.ypr[0],
            self.low_state.imu_state.ypr[1],
            self.low_state.imu_state.ypr[2]
        )
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        Rb_w = rot.as_matrix()
        rpy = np.array([self.low_state.imu_state.ypr[2], 
                       self.low_state.imu_state.ypr[1], 
                       self.low_state.imu_state.ypr[0]])
        R_xyz_omega = np.eye(3)
        R_xyz_omega[1, :] = R.from_euler('x', rpy[0]).as_matrix()[1, :]
        R_xyz_omega[2, :] = (R.from_euler('x', rpy[0]) * R.from_euler('y', rpy[1])).as_matrix()[2, :]
        
        # Get linear velocity (simplified, may need actual velocity estimation)
        lin_vel_ = 0.0  # Placeholder, should be computed from actual velocity
        
        if self.gait_d == "Walk" and norm_cmd < 0.1:
            if lin_vel_ > 0.1:
                self.gait_d = "Walk"
                self.phase_counter = 1
            else:
                self.rl_counter = self.phase_counter = self.stand_counter
                self.gait_d = "Stand"
        elif self.gait_d == "Stand":
            self.rl_counter = self.phase_counter = self.stand_counter
            self.gait_d = "Stand"
        
        if self.gait_d == "Stand" and norm_cmd > 0.1:
            self.phase_counter = 1
            self.gait_d = "Walk"
        elif self.gait_d == "Walk" and norm_cmd > 0.1:
            self.phase_counter = 1
            self.gait_d = "Walk"
        
        self.gait_a = self.gait_d

    def run(self):
        """Main run loop with command processing and action execution"""
        if self.config.msg_type != "adam_lite":
            for i in range(12):
                self.hand_cmd.position[i] = self.close_hand[i]
            self.hand_pub.Write(self.hand_cmd)
        
        # Process joystick commands
        # Note: offset methods may not exist in RemoteController, using 0.0 as default
        # self.x_vel_command_offset += self.remote_controller.get_walk_x_direction_speed_offset()
        # self.y_vel_command_offset += self.remote_controller.get_walk_y_direction_speed_offset()
        
        self.joystick_command[0] = self.remote_controller.get_walk_x_direction_speed()
        self.joystick_command[1] = self.remote_controller.get_walk_y_direction_speed()
        self.joystick_command[2] = self.remote_controller.get_walk_yaw_direction_speed()
        
        self.joystick_command[0] += self.x_vel_command_offset
        self.joystick_command[1] += self.y_vel_command_offset
        
        # Smooth command transitions
        if abs(self.joystick_command[0]) > 0.05 or abs(self.joystick_command[1]) > 0.02 or abs(self.joystick_command[2]) > 0.02:
            # X velocity command smoothing
            if abs(self.joystick_command[0]) > 0.1:
                if abs(self.joystick_command[0] - self.cmd[0]) > 0.0009:
                    self.cmd[0] += 0.0009 * np.sign(self.joystick_command[0] - self.cmd[0])
                else:
                    self.cmd[0] = self.joystick_command[0]
            else:
                if abs(self.joystick_command[0] - self.cmd[0]) > 0.0015:
                    self.cmd[0] += 0.0015 * np.sign(self.joystick_command[0] - self.cmd[0])
                else:
                    self.cmd[0] = self.joystick_command[0]
            self.cmd[0] = np.clip(self.cmd[0], -0.5, 1.5)
            
            # Y velocity command
            self.cmd[1] = self.joystick_command[1]
            self.cmd[1] = np.clip(self.cmd[1], -0.3, 0.3)
            
            # Yaw velocity command smoothing
            if abs(self.joystick_command[2] - self.cmd[2]) > 0.001 and abs(self.joystick_command[2]) > 0.1:
                self.cmd[2] += 0.001 * np.sign(self.joystick_command[2] - self.cmd[2])
            else:
                self.cmd[2] = self.joystick_command[2]
            self.cmd[2] = np.clip(self.cmd[2], -1.0, 1.0)
            
            self.command_ori = self.cmd.copy()
            self.pos_percent = 0.0
        elif self.gait_d == "Walk" or self.pos_percent < 1.0:
            self.cmd[0] = self.command_ori[0] * (1 - self.pos_percent)
            self.cmd[1] = self.command_ori[1] * (1 - self.pos_percent)
            self.cmd[2] = self.command_ori[2] * (1 - self.pos_percent)
            self.pos_percent += 1.0 / self.pos_duration
            self.pos_percent = min(self.pos_percent, 1.0)
        
        # Compute observations and actions
        self.computeObs()
        self.computeActions()
        self.changeState()
        
        # Update action_last
        self.action_last[:len(self.output_data_mlp)] = self.output_data_mlp.copy()
        
        # Compute smoothed action output
        mlp_out = self.output_data_mlp.copy()
        mlp_out_dot = np.zeros_like(self.output_data_mlp)
        
        if self.gait_a == "Walk":
            self.timer_plan += self.config.control_dt
            mlp_out = self.para_0 + self.para_1 * self.timer_plan + \
                     self.para_2 * self.timer_plan**2 + \
                     self.para_3 * self.timer_plan**3
            mlp_out_dot = self.para_1 + 2.0 * self.para_2 * self.timer_plan + \
                         3.0 * self.para_3 * self.timer_plan**2
        
        # Scale action to joint positions
        mlp_out_scaled = mlp_out * self.action_scales + self.config.default_angles
        mlp_out_dot_scaled = mlp_out_dot * self.action_scales
        
        # Do not control ankle roll (keep current position)
        if len(self.config.leg_joint2motor_idx) >= 12:
            mlp_out_scaled[5] = self.low_state.motor_state[5].q
            mlp_out_scaled[11] = self.low_state.motor_state[11].q
        
        # Build low cmd
        num_leg_joints = len(self.config.leg_joint2motor_idx)
        for i in range(num_leg_joints):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = mlp_out_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # Update last action states
        self.last_action_d = mlp_out.copy()
        self.last_action_dot_d = mlp_out_dot.copy()
        
        # Send command
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


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
            if controller.remote_controller.button[KeyMap.select] == 1:
                controller.move_to_default_pos()
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
