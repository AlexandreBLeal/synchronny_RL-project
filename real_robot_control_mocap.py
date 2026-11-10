#!/usr/bin/env python3
"""
Object sequencing:
  obj_1 → obj_2 → obj_3 → obj_4
  Each transition fires when the EEF reaches the current target.
  When obj_4 reaches its target the episode ends.

terminal 1:  roscore
terminal 2:  roslaunch mocap_optitrack mocap.launch
"""

import sys
import time
import threading

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from xarm.wrapper import XArmAPI

from evalute_trainning import Lite6PickPlaceEnv
import pybullet as p

#from playsound import playsound
import sounddevice as sd

def bip(freq, dur, a=0, d=0, s=1, r=0):
    t = np.arange(0, dur, 1/44100)
    env = np.interp(t, [0, a, a+d, dur-r, dur], [0, 1, s, s, 0])
    sound = np.sin(2*np.pi*freq*t) * env
    sd.play(sound.astype(np.float32), samplerate=44100, blocksize=2048)
    sd.wait()  # wait until done before continuing

# At the top of the file, after MAX_CONSECUTIVE_FAULTS:
MANUAL_GRIPPER_CONTROL = True   # True = use distance-based drop; False = use model output
GRIPPER_DROP_DISTANCE  = 0.07   # metres from target to release the object
#csv_filename = "real_test11.csv"
csv_filename = "constant_test9.csv"

# ---------------------------------------------------------------------------
# UFactory Lite 6 joint limits (radians) — from the official spec sheet
# ---------------------------------------------------------------------------
LITE6_JOINT_LIMITS_LOW  = np.array([-6.2832, -2.0944, -0.0873, -6.2832, -1.6929, -6.2832])
LITE6_JOINT_LIMITS_HIGH = np.array([ 6.2832,  2.6180,  5.3127,  6.2832,  3.8397,  6.2832])

# Safety margins applied on top of hardware limits (radians)
SAFETY_MARGIN = np.deg2rad(2.0)  # 2° buffer inside each limit

SAFE_LOW  = LITE6_JOINT_LIMITS_LOW  + SAFETY_MARGIN
SAFE_HIGH = LITE6_JOINT_LIMITS_HIGH - SAFETY_MARGIN

# Max joint displacement allowed in a single 1/24 s step (radians)
# 0.08 rad ≈ 4.6° per step  →  ~110°/s max, comfortable for Lite 6
MAX_DELTA_PER_STEP = 0.08

# Servo-mode motion parameters
SERVO_VELO = 1.5           # rad/s  (kept from original, well within limits)
SERVO_ACC  = np.deg2rad(500)  # rad/s²  (≈ 8.73, same as original)

# How many consecutive faults before we abort the episode
MAX_CONSECUTIVE_FAULTS = 5

# Error codes we consider recoverable
RECOVERABLE_CODES = {1, 9}   # not-ready / motion-plan failure
FATAL_CODES       = {22}      # joint limit violation (needs manual check)

# xArm SDK error-code descriptions (subset relevant to Lite 6 servo mode)
ERROR_DESCRIPTIONS = {
    1:  "Controller not ready / not connected",
    2:  "Emergency stop active",
    9:  "Motion plan failed (unreachable config or collision)",
    11: "Overspeed",
    19: "Inverse kinematics no solution",
    22: "Joint angle out of range",
    23: "Speed out of range",
    24: "Acceleration out of range",
    35: "Safety boundary violation",
}

# ---------------------------------------------------------------------------
# Object sequencing
#   The active object index advances 1 → 2 → 3 → 4 as the EEF reaches each
#   successive target.  Index 4 is the final object; reaching its target ends
#   the episode.
# ---------------------------------------------------------------------------
REACH_THRESHOLD  = 0.05   # 5 cm
NUM_OBJECTS      = 4      # total objects in the chain

# Active object index (1-based).  Guarded by _obj_index_lock.
_active_obj_index      = 1
_active_obj_index_lock = threading.Lock()

# Set to True by the control loop when obj_4 has reached its target.
_episode_complete      = False
_episode_complete_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Mocap globals — one slot per object
# ---------------------------------------------------------------------------
robot2_pose      = None
robot2_pose_lock = threading.Lock()

# obj_poses[i] holds the latest PoseStamped data for obj_{i+1}  (0-indexed)
obj_poses      = [None, None, None, None]
obj_poses_lock = threading.Lock()

# ---------------------------------------------------------------------------
# CSV Logging
# ---------------------------------------------------------------------------
import csv
import os
from datetime import datetime

_csv_file   = None
_csv_writer = None

def init_csv_logger(path=None) -> str:
    """
    Open a CSV file for trajectory logging.
    Returns the path used.
    """
    global _csv_file, _csv_writer

    if path is None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"trajectory_{ts}.csv"

    _csv_file = open(path, "w", newline="")
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow([
        "timestamp",
        "eef_x", "eef_y", "eef_z",
        "eef_vel_x", "eef_vel_y", "eef_vel_z",
        "human_x", "human_y", "human_z",
        "human_vel_x", "human_vel_y", "human_vel_z",
        "obj_x", "obj_y", "obj_z",
        "active_obj_index",
    ])
    _csv_file.flush()
    rospy.loginfo("CSV logger opened: %s", path)
    return path


def log_state(
    eef_pos,
    eef_vel,
    human_pos,
    human_vel,
    obj_pos,
    active_obj_idx: int,
) -> None:
    """
    Append one row to the open CSV log.
    Silently skips if the logger was never initialised or any array is None.
    """
    if _csv_writer is None:
        return
    if any(x is None for x in (eef_pos, eef_vel, human_pos, human_vel, obj_pos)):
        return

    _csv_writer.writerow([
        time.time(),
        *np.round(eef_pos,   6),
        *np.round(eef_vel,   6),
        *np.round(human_pos, 6),
        *np.round(human_vel, 6),
        *np.round(obj_pos,   6),
        active_obj_idx,
    ])
    _csv_file.flush()


def close_csv_logger() -> None:
    """Flush and close the CSV file."""
    global _csv_file, _csv_writer
    if _csv_file is not None:
        _csv_file.flush()
        _csv_file.close()
        _csv_file   = None
        _csv_writer = None
        rospy.loginfo("CSV logger closed.")
# ---------------------------------------------------------------------------
# Geometry helpers  (unchanged from original)
# ---------------------------------------------------------------------------
def rotation_matrix_3d(roll, pitch, yaw):
    r, p, y = np.radians(roll), np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]])
    return Rz @ Ry @ Rx


class FrameTransform:
    def __init__(self, R, t):
        self.R = np.array(R, dtype=float)
        self.t = np.array(t, dtype=float)

    @classmethod
    def from_3d(cls, roll, pitch, yaw, tx=0.0, ty=0.0, tz=0.0):
        return cls(rotation_matrix_3d(roll, pitch, yaw), [tx, ty, tz])

    def transform(self, points):
        points = np.atleast_2d(points)
        return (self.R @ points.T).T + self.t


# ---------------------------------------------------------------------------
# Mocap callbacks
# ---------------------------------------------------------------------------
def _pack_pose(msg):
    return {
        "position":    {"x": msg.pose.position.x,
                        "y": msg.pose.position.y,
                        "z": msg.pose.position.z},
        "orientation": {"x": msg.pose.orientation.x,
                        "y": msg.pose.orientation.y,
                        "z": msg.pose.orientation.z,
                        "w": msg.pose.orientation.w},
        "timestamp":   msg.header.stamp.to_sec(),
    }

def mocap_callback(msg):
    global robot2_pose
    with robot2_pose_lock:
        robot2_pose = _pack_pose(msg)

def _make_obj_callback(idx):
    """Return a ROS subscriber callback that writes into obj_poses[idx]."""
    def cb(msg):
        with obj_poses_lock:
            obj_poses[idx] = _pack_pose(msg)
    return cb

def get_robot2_pose():
    with robot2_pose_lock:
        return dict(robot2_pose) if robot2_pose is not None else None

def get_obj_pose(obj_number):
    """Return the latest pose for obj_{obj_number} (1-based), or None."""
    idx = obj_number - 1
    with obj_poses_lock:
        p = obj_poses[idx]
        return dict(p) if p is not None else None

def get_active_obj_pose():
    """Return the pose for the currently active object."""
    with _active_obj_index_lock:
        idx = _active_obj_index
    return get_obj_pose(idx)


def check_and_advance_obj_source(eef_pos, target_pos):
    """
    Compare EEF position to the current target.  If close enough, advance the
    active object index by one (1→2, 2→3, 3→4).

    When the index would advance past NUM_OBJECTS (i.e. obj_4 was just
    reached), set _episode_complete instead and return 4 to signal the
    control loop to end.

    Return values
    -------------
    0  — not close enough yet, no change
    N  — (1-based) the NEW active object index after the advance
    4  — obj_4 just reached its target; episode should end
    """
    global _active_obj_index, _episode_complete

    with _active_obj_index_lock:
        current_idx = _active_obj_index

        dist = float(np.linalg.norm(np.array(eef_pos) - np.array(target_pos)))
        if dist >= REACH_THRESHOLD:
            return 0  # not there yet

        # ── EEF has reached the target ──────────────────────────────────────
        next_idx = current_idx + 1

        if next_idx > NUM_OBJECTS:
            # obj_4 target reached → episode complete
            with _episode_complete_lock:
                _episode_complete = True
            rospy.loginfo(
                "EEF reached obj_%d target (dist=%.4f m) — all objects handled, "
                "ending episode.", current_idx, dist,
            )
            return NUM_OBJECTS  # sentinel: episode done

        # Advance to the next object
        _active_obj_index = next_idx
        rospy.loginfo(
            "EEF reached obj_%d target (dist=%.4f m < %.4f m) — "
            "switching active object source: obj_%d → obj_%d",
            current_idx, dist, REACH_THRESHOLD, current_idx, next_idx,
        )
        return next_idx


def is_episode_complete():
    with _episode_complete_lock:
        return _episode_complete


def reset_obj_sequencing():
    """Call at the start of each episode to reset sequencing state."""
    global _active_obj_index, _episode_complete
    with _active_obj_index_lock:
        _active_obj_index = 1
    with _episode_complete_lock:
        _episode_complete = False


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------
def clamp_joints(angles):
    """Hard-clamp joint angles to safe operating range."""
    return np.clip(np.array(angles, dtype=float), SAFE_LOW, SAFE_HIGH)


def cap_delta(target, prev, max_delta=MAX_DELTA_PER_STEP):
    """
    Limit the step size for each joint independently.
    Returns the capped target and a flag indicating whether capping occurred.
    """
    delta   = np.array(target) - np.array(prev)
    clamped = np.clip(delta, -max_delta, max_delta)
    capped  = np.array(prev) + clamped
    was_capped = not np.allclose(delta, clamped, atol=1e-6)
    return capped, was_capped


def describe_error(code):
    return ERROR_DESCRIPTIONS.get(code, f"Unknown error code {code}")


def recover_arm(arm, context=""):
    """
    Attempt to clear faults and re-enable the arm.
    Returns True if recovery succeeded, False otherwise.
    """
    rospy.logwarn("Fault recovery attempt%s ...", f" [{context}]" if context else "")

    # 1. Clear any error / warning
    arm.clean_error()
    arm.clean_warn()
    time.sleep(0.1)

    # 2. Re-enable motion
    ret = arm.motion_enable(enable=True)
    if ret != 0:
        rospy.logerr("motion_enable failed after recovery (ret=%d)", ret)
        return False

    # 3. Re-enter servo mode
    arm.set_mode(1)
    time.sleep(0.05)
    arm.set_state(0)
    time.sleep(0.1)

    # 4. Confirm error is gone
    if arm.error_code != 0:
        rospy.logerr(
            "Recovery failed — error_code still %d (%s)",
            arm.error_code,
            describe_error(arm.error_code),
        )
        return False

    rospy.loginfo("Recovery successful.")
    return True


def safe_set_gripper(arm, on: bool, current_state: bool):
    """
    Send gripper command only when state actually changes.
    Returns the new gripper state.
    """
    if on == current_state:
        return current_state

    ret = arm.set_vacuum_gripper(on=on, wait=False)
    if ret != 0:
        rospy.logwarn("set_vacuum_gripper(on=%s) returned code=%d", on, ret)
    return on  # optimistically update; fault check happens in the main loop


def safe_send_joints(arm, target_angles, prev_angles):
    """
    1. Cap per-step delta
    2. Clamp to joint limits
    3. Send via set_servo_angle_j with speed/acc caps
    4. Return (sent_angles, ok)
    """
    capped, was_capped = cap_delta(target_angles, prev_angles)
    if was_capped:
        rospy.logwarn(
            "Joint delta capped. Max raw delta=%.4f rad",
            float(np.max(np.abs(np.array(target_angles) - np.array(prev_angles)))),
        )

    safe = clamp_joints(capped)
    if not np.allclose(safe, capped, atol=1e-6):
        rospy.logwarn(
            "Joint angle clamped to safe limits. Diff: %s rad",
            np.round(safe - capped, 4),
        )

    # Skip command if negligible motion (avoids spamming controller)
    if np.max(np.abs(safe - np.array(prev_angles))) < 0.002:
        return list(safe), True

    ret = arm.set_servo_angle_j(
        angles=list(safe),
        is_radian=True,
        speed=SERVO_VELO,
        acc=SERVO_ACC,
    )

    if ret not in (0, None):
        rospy.logwarn(
            "set_servo_angle_j returned code=%d (%s)", ret, describe_error(ret)
        )
        return list(safe), False

    return list(safe), True


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
def get_real_obs(obs, prev_eef_pos, prev_obj_pos, human_pos_history, prev_eef_vel, step_i, verbose=False):
    pose     = get_robot2_pose()
    obj_pose = get_active_obj_pose()

    if pose is None or obj_pose is None:
        rospy.logwarn("Step %d | Missing mocap data — reusing previous observation.", step_i)
        return obs, prev_eef_pos, prev_obj_pos, human_pos_history, prev_eef_vel, 0

    T_3d = FrameTransform.from_3d(
        roll=0, pitch=0, yaw=90,
        tx=0.97 - 0.059, ty=-0.337, tz=-0.8 + 0.033,
    )

    obj_trans_pos   = T_3d.transform(np.array([
        obj_pose["position"]["x"],
        obj_pose["position"]["y"],
        obj_pose["position"]["z"],
    ]))
    human_trans_pos = T_3d.transform(np.array([
        pose["position"]["x"],
        pose["position"]["y"],
        pose["position"]["z"],
    ]))
    human_pos = human_trans_pos[0]

    if verbose:
        rospy.loginfo("Step %d | obj_trans: %s", step_i, np.round(obj_trans_pos[0], 4))
        rospy.loginfo("Step %d | human_trans: %s", step_i, np.round(human_pos, 4))

    eef_pos    = obs[0, :3]
    target_pos = obs[0, 22:25]

    # Check whether to advance the active object source (returns new index or 0)
    change_object = check_and_advance_obj_source(obj_trans_pos[0], target_pos)

    prev_human_pos   = human_pos_history[-1]
    relative_obj_pos = obj_trans_pos[0] - eef_pos
    obj_vel          = (prev_obj_pos - obj_trans_pos[0]) / (1.0 / 24.0)
    eef_vel          = (eef_pos - prev_eef_pos)          / (1.0 / 24.0)
    relative_obj_vel = obj_vel - eef_vel

    human_pos_history        = np.roll(human_pos_history, shift=-1, axis=0)
    human_pos_history[-1]    = human_pos
    human_vel                = (human_pos - prev_human_pos) / (1.0 / 24.0)

    #obs[0, 3:6]  = obj_trans_pos[0]
    #obs[0, 6:9]  = relative_obj_pos
    #obs[0, 13:16] = relative_obj_vel
    obs[0, 29:44] = human_pos_history.flatten()
    obs[0, 44:47] = human_vel

    return (
        obs,
        eef_pos.copy(),
        obj_trans_pos[0].copy(),
        human_pos_history,
        eef_vel.copy(),
        change_object,
    )


def next_obj_pose():
    obj_pose = get_active_obj_pose()
    if obj_pose is None:
        return None

    T_3d = FrameTransform.from_3d(
        roll=0, pitch=0, yaw=90,
        tx=0.97 - 0.059, ty=-0.337, tz=-0.8 + 0.033,
    )

    obj_trans_pos = T_3d.transform(np.array([
        obj_pose["position"]["x"],
        obj_pose["position"]["y"],
        obj_pose["position"]["z"],
    ]))
    return obj_trans_pos[0]


# ---------------------------------------------------------------------------
# Per-object drop target positions (robot frame).
# obj_1 → obj_2 uses the original target; extend for 3 and 4 as needed.
# ---------------------------------------------------------------------------
# Maps: new_active_index → target_pos to assign when we switch TO that object
OBJ_DROP_TARGETS = {
    2: np.array([0.25, -0.07, 0.06]),   # target when obj_2 becomes active
    3: np.array([0.33, -0.14, 0.06]),   # ← update with real coords for obj_3
    4: np.array([0.33, -0.07, 0.06]),   # ← update with real coords for obj_4
}


# ---------------------------------------------------------------------------
# Arm initialisation helpers
# ---------------------------------------------------------------------------
def print_info(arm):
    print("position:",        arm.position)
    print("tcp_load:",        arm.tcp_load)
    print("tcp_offset:",      arm.tcp_offset)
    print("tcp_speed_limit:", arm.tcp_speed_limit)
    print("===== JOINT =====")
    print("angles:",           arm.angles)
    print("joint_speed_limit:", arm.joint_speed_limit)


def set_initial_pos(arm):
    arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, speed=100, wait=True)


def init_arm(ip="192.168.1.194"):
    """Connect, clear faults, home, and return arm handle."""
    arm = XArmAPI(ip)
    time.sleep(0.5)

    if arm.warn_code  != 0: arm.clean_warn()
    if arm.error_code != 0: arm.clean_error()

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)

    arm.set_vacuum_gripper(on=False)
    set_initial_pos(arm)
    time.sleep(1)

    return arm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model_path   = "ppo_lite6_pick_place_ppo_final"
    vecnorm_path = "ppo_vec_normalize.pkl"
    verbose      = False

    # ---- ROS setup ----
    rospy.init_node("robot_control_node", anonymous=True)
    rospy.Subscriber("/mocap_node/Robot_2/pose", PoseStamped, mocap_callback)

    # Subscribe to all four object topics dynamically
    for _n in range(1, NUM_OBJECTS + 1):
        rospy.Subscriber(
            f"/mocap_node/obj_{_n}/pose",
            PoseStamped,
            _make_obj_callback(_n - 1),   # writes to obj_poses[_n-1]
        )

    rospy.loginfo("Waiting for mocap data …")

    timeout = 10.0
    start_t = time.time()
    while True:
        with obj_poses_lock:
            all_received = all(p is not None for p in obj_poses)
        if get_robot2_pose() is not None and all_received:
            break
        if time.time() - start_t > timeout:
            rospy.logwarn("No mocap data after %.1f s — proceeding anyway.", timeout)
            break
        time.sleep(0.1)

    # ---- State ----
    human_pos_history = np.zeros((5, 3), dtype=np.float32)
    prev_eef_pos      = np.zeros(3)
    prev_obj_pos      = np.zeros(3)
    prev_eef_vel      = np.zeros(3)

    # ---- RL ----
    env   = DummyVecEnv([lambda: Lite6PickPlaceEnv(render_mode="real", eval_mode=True)])
    model = PPO.load(model_path, env=env)
    obs   = env.reset()

    # ---- Arm ----
    arm = init_arm("192.168.1.194")

    prev_joint_poses    = None
    gripper_state       = False
    consecutive_faults  = 0

    reset_obj_sequencing()   # ensure clean state at episode start
    init_csv_logger(csv_filename) 

    #starting sound
    # Example
    # A banal computer bip
    #bip(880, 0.1)

    # With a more sophisticated envelope. Djdoing
    #bip(880,0.4, a=0, d=0.24, s=0.28, r=0.09)
    
    # ---- Control loop ----
    for i in range(1400):
        step_start = time.time()

        # ── Early exit: obj_4 target already confirmed in previous step ──────
        if is_episode_complete():
            rospy.loginfo("All objects handled — ending episode at step %d.", i)
            break

        # ── Observation ──────────────────────────────────────────────────────
        obs, prev_eef_pos, prev_obj_pos, human_pos_history, prev_eef_vel, change_object = get_real_obs(
            obs, prev_eef_pos, prev_obj_pos, human_pos_history, prev_eef_vel,
            step_i=i, verbose=verbose,
        )

        # ── Sync PyBullet object position from mocap ──────────────────────────
        obj_mocap = next_obj_pose()
        #print("obj_mocap, change_object ",obj_mocap, change_object)
        if obj_mocap is not None:
            raw_env = env.envs[0]

            if raw_env._trajectory is None or change_object != 0 or raw_env._traj_index == 0:
                p.resetBasePositionAndOrientation(
                    raw_env._object_id,
                    [obj_mocap[0], obj_mocap[1], obj_mocap[2]],
                    [0, 0, 0, 1],
                    physicsClientId=raw_env._physics_client,
                )
                if raw_env._grasped:
                    p.removeConstraint(raw_env._grasp_constraint,
                                       physicsClientId=raw_env._physics_client)
                    raw_env._grasped = False

                # On a successful advance (change_object is the new index),
                # update the environment's target position accordingly.
                if change_object in OBJ_DROP_TARGETS:
                    raw_env._target_pos = OBJ_DROP_TARGETS[change_object].copy()
                    rospy.loginfo(
                        "Updated env target for obj_%d → %s",
                        change_object, raw_env._target_pos,
                    )

                # If obj_4 just reached its target the episode is over;
                # is_episode_complete() will catch it at the top of next iter.
                if change_object == NUM_OBJECTS and is_episode_complete():
                    rospy.loginfo("obj_4 reached target — breaking after sync.")
                    break

                current_eef = raw_env._get_eef_pos()
                raw_env._build_trajectory(current_eef)

        # ── Policy ───────────────────────────────────────────────────────────
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)

        # ── First step: move to sim pose in position mode, then switch modes ─
        if i == 0:
            print_info(arm)
            curr = list(info[0]["joint_poses"][:6])
            curr_safe = list(clamp_joints(curr))
            arm.set_servo_angle(angle=curr_safe, wait=True, speed=1.5, is_radian=True)
            prev_joint_poses = curr_safe

            arm.set_mode(1)   # servo / streaming mode
            arm.set_state(0)
            time.sleep(5)
            gripper_state = False
            # starting sound
            bip(440,0.6, a=0.1, d=0.2, s=0.5, r=0.2)
            continue         # skip the rest of iteration 0

        # ── Gripper ──────────────────────────────────────────────────────────
        
        # ── Gripper ──────────────────────────────────────────────────────────
        if MANUAL_GRIPPER_CONTROL:
            obj_mocap_pos = next_obj_pose()
            target_pos    = obs[0, 22:25]
            if obj_mocap_pos is not None:
                dist_to_target = float(np.linalg.norm(obj_mocap_pos - target_pos))
                want_gripper   = dist_to_target > GRIPPER_DROP_DISTANCE -0.02
            else:
                want_gripper = gripper_state   # hold current state if no data
        else:
            try:
                want_gripper = action[0][3] > 0
            except Exception:
                want_gripper = action[0][1] > 0

        gripper_state = safe_set_gripper(arm, want_gripper, gripper_state)
        
        # ── Joint command ────────────────────────────────────────────────────
        target_joints = list(info[0]["joint_poses"])[:6]
        sent_joints, cmd_ok = safe_send_joints(arm, target_joints, prev_joint_poses)

        if cmd_ok:
            prev_joint_poses   = sent_joints
            consecutive_faults = 0
        else:
            consecutive_faults += 1
            rospy.logwarn(
                "Command failed (fault %d/%d). arm.error_code=%d (%s)",
                consecutive_faults, MAX_CONSECUTIVE_FAULTS,
                arm.error_code, describe_error(arm.error_code),
            )

        # ── Fault handling ───────────────────────────────────────────────────
        if arm.error_code != 0 or arm.warn_code != 0:
            code = arm.error_code
            rospy.logwarn(
                "Arm fault detected: error_code=%d (%s), warn_code=%d",
                code, describe_error(code), arm.warn_code,
            )

            if code in FATAL_CODES:
                rospy.logerr(
                    "Fatal error code %d — stopping episode for safety.", code
                )
                break

            recovered = recover_arm(arm, context=f"step={i}, code={code}")
            if not recovered:
                consecutive_faults += 1
            else:
                consecutive_faults = 0
                arm.set_mode(1)
                arm.set_state(0)
                time.sleep(0.05)

        if consecutive_faults >= MAX_CONSECUTIVE_FAULTS:
            rospy.logerr(
                "%d consecutive faults — aborting episode.", consecutive_faults
            )
            break
        # ── Log state ────────────────────────────────────────────────────────
        with _active_obj_index_lock:
            _log_obj_idx = _active_obj_index
        log_state(
            eef_pos       = prev_eef_pos,
            eef_vel       = prev_eef_vel,
            human_pos     = human_pos_history[-1],
            human_vel     = obs[0, 44:47],
            obj_pos       = prev_obj_pos,
            active_obj_idx= _log_obj_idx,
        )

        # ── Timing: hold 1/24 s step ─────────────────────────────────────────
        elapsed   = time.time() - step_start
        remaining = (1.0 / 24.0) - elapsed
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -0.005:
            rospy.logwarn("Step %d overran budget by %.1f ms", i, -remaining * 1000)

        # ── RL episode end (environment signals done) ─────────────────────────
        if done[0]:
            rospy.loginfo("RL episode done at step %d.", i)
            reset_obj_sequencing()
            obs = env.reset()
            break

    # ── Shutdown ──────────────────────────────────────────────────────────────
    rospy.loginfo("Shutting down …")
    close_csv_logger()   
    arm.set_vacuum_gripper(on=False)
    time.sleep(0.5)

    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    set_initial_pos(arm)
    time.sleep(1)

    arm.disconnect()
    env.close()