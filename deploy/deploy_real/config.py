from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            self.estimatory_path = None
            if "estimatory_path" in config:
                self.estimatory_path = config["estimatory_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.history_length = config.get("history_length", 1)

            _ids_map = config.get("joint_ids_map", None)
            if _ids_map is not None:
                self.joint_ids_map = np.array(_ids_map, dtype=np.int32)
            else:
                self.joint_ids_map = np.arange(self.num_actions, dtype=np.int32)

            # Optional parameters for advanced configurations
            self.delta_num = None
            if "delta_num" in config:
                self.delta_num = config["delta_num"]

            self.frame_stack = None
            if "frame_stack" in config:
                self.frame_stack = config["frame_stack"]

            self.latent_frame_stack = None
            if "latent_frame_stack" in config:
                self.latent_frame_stack = config["latent_frame_stack"]

            self.latent_size = None
            if "latent_size" in config:
                self.latent_size = config["latent_size"]

            self.cycle_time_stand = None
            if "cycle_time_stand" in config:
                self.cycle_time_stand = config["cycle_time_stand"]

            self.cycle_time_walk = None
            if "cycle_time_walk" in config:
                self.cycle_time_walk = config["cycle_time_walk"]
