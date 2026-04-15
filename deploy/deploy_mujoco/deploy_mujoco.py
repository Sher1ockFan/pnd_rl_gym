import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import yaml
import collections
import math

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None


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


class ObsBuffer:
    """Ring buffer that keeps *history_length* frames for one observation term."""
    def __init__(self, dim, history_length=1):
        self.dim = dim
        self.history_length = history_length
        self._buf = collections.deque(maxlen=history_length)
        self.reset()

    def reset(self):
        self._buf.clear()
        for _ in range(self.history_length):
            self._buf.append(np.zeros(self.dim, dtype=np.float32))

    def push(self, obs):
        self._buf.append(obs.astype(np.float32))

    def flat(self):
        return np.concatenate(list(self._buf))


def load_policy(path):
    if path.endswith(".onnx"):
        assert ort is not None, "onnxruntime is required for .onnx policy"
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        def _infer(obs_np):
            return sess.run([out_name], {in_name: obs_np.reshape(1, -1).astype(np.float32)})[0].squeeze()
        return _infer
    else:
        assert torch is not None, "torch is required for .pt/.jit policy"
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        def _infer(obs_np):
            with torch.no_grad():
                return model(torch.from_numpy(obs_np).unsqueeze(0).float()).squeeze().numpy()
        return _infer


if __name__ == "__main__":
    # get config file name from command line
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

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # ---- auto-detect history_length & gait_phase from num_obs ----
    gait_period = config.get("gait_period", 0.8)
    base_no_phase = 3 * num_actions + 9
    base_with_phase = base_no_phase + 2

    history_length = 1
    use_gait_phase = False

    if num_obs % base_with_phase == 0:
        use_gait_phase = True
        history_length = num_obs // base_with_phase
    elif num_obs % base_no_phase == 0:
        use_gait_phase = False
        history_length = num_obs // base_no_phase
    else:
        raise ValueError(
            f"num_obs={num_obs} cannot be decomposed with num_actions={num_actions}. "
            f"Expected multiple of {base_no_phase} or {base_with_phase}."
        )

    # ---- build observation history buffers ----
    buf_ang_vel = ObsBuffer(3, history_length)
    buf_gravity = ObsBuffer(3, history_length)
    buf_cmd     = ObsBuffer(3, history_length)
    buf_pos     = ObsBuffer(num_actions, history_length)
    buf_vel     = ObsBuffer(num_actions, history_length)
    buf_action  = ObsBuffer(num_actions, history_length)
    obs_buffers = [buf_ang_vel, buf_gravity, buf_cmd, buf_pos, buf_vel, buf_action]

    if use_gait_phase:
        buf_phase = ObsBuffer(2, history_length)
        obs_buffers.append(buf_phase)

    total_obs = sum(b.dim * b.history_length for b in obs_buffers)
    assert total_obs == num_obs, f"obs mismatch: computed {total_obs} != config {num_obs}"
    print(f"[deploy] obs_dim={total_obs}  history={history_length}  gait_phase={use_gait_phase}")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = load_policy(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # Only control the first num_actions joints (e.g., 12 for legs only)
            tau = pd_control(target_dof_pos, d.qpos[7:7+num_actions], kps, np.zeros_like(kds), d.qvel[6:6+num_actions], kds)
            d.ctrl[:num_actions] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation (only for controlled joints)
                qj = d.qpos[7:7+num_actions]
                dqj = d.qvel[6:6+num_actions]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                # push into history buffers (scaled)
                buf_ang_vel.push(omega * ang_vel_scale)
                buf_gravity.push(get_gravity_orientation(quat))
                buf_cmd.push(cmd * cmd_scale)
                buf_pos.push((qj - default_angles) * dof_pos_scale)
                buf_vel.push(dqj * dof_vel_scale)
                buf_action.push(action)

                if use_gait_phase:
                    count = counter * simulation_dt
                    phase = count % gait_period / gait_period
                    sin_phase = np.sin(2 * np.pi * phase)
                    cos_phase = np.cos(2 * np.pi * phase)
                    buf_phase.push(np.array([sin_phase, cos_phase], dtype=np.float32))

                obs = np.concatenate([b.flat() for b in obs_buffers])

                # policy inference
                action = policy(obs)
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
