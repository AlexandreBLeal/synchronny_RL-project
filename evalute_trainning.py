import pandas as pd
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import time, random
import math
import csv  # Added for robust, dynamic header parsing


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_3d(roll, pitch, yaw):
    r, p_, y = np.radians(roll), np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(r), -np.sin(r)],
                   [0,  np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p_), 0, np.sin(p_)],
                   [0,           1,           0],
                   [-np.sin(p_), 0, np.cos(p_)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,          1]])
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

    def inverse(self):
        R_inv = self.R.T
        return FrameTransform(R_inv, -R_inv @ self.t)

    def compose(self, other):
        return FrameTransform(self.R @ other.R, self.R @ other.t + self.t)

    def as_homogeneous(self):
        dim = len(self.t)
        H = np.eye(dim + 1)
        H[:dim, :dim] = self.R
        H[:dim,  dim] = self.t
        return H

    def __repr__(self):
        return f"FrameTransform(\n  R={self.R},\n  t={self.t}\n)"


# ─────────────────────────────────────────────────────────────────────────────
# Minimum-jerk (5th-degree polynomial) trajectory
# ─────────────────────────────────────────────────────────────────────────────

def min_jerk_traj(p_start: np.ndarray,
                  p_end:   np.ndarray,
                  point_spacing: float = 0.002) -> np.ndarray:
    """
    Generate a minimum-jerk (5th-order polynomial) trajectory between two 3-D
    points with approximately `point_spacing` metres between consecutive samples.

    Returns
    -------
    waypoints : np.ndarray  shape (N, 3)
        The first point equals p_start; the last equals p_end.
    """
    dist = np.linalg.norm(p_end - p_start)
    if dist < 1e-6:
        return np.array([p_start.copy(), p_end.copy()])

    N = max(2, int(np.ceil(dist / point_spacing)) + 1)
    tau = np.linspace(0.0, 1.0, N)

    # Minimum-jerk blend: s(τ) = 10τ³ − 15τ⁴ + 6τ⁵
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

    waypoints = p_start[None, :] + s[:, None] * (p_end - p_start)[None, :]
    return waypoints.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class Lite6PickPlaceEnv(gym.Env):
    """
    Gym environment for the uFactory Lite 6 with vacuum gripper.

    Trajectory-based control
    ────────────────────────
    At every reset (and after each successful pick-place cycle) the environment
    pre-computes a **four-segment** minimum-jerk trajectory that guarantees a
    top-down approach to the object and a top-down descent onto the target:

        Seg 0  (lateral approach) : current EEF pos  → above object  (cruise_height)
        Seg 1  (descend to pick)  : above object      → pick pos      (pick_height)
        Seg 2  (lateral transfer) : pick pos           → above target  (cruise_height)
        Seg 3  (descend to place) : above target       → target pos

    The agent controls:
        • speed   ∈ [-1, 1]  — how many trajectory steps to advance per timestep
                                 (mapped to ±MAX_TRAJ_STEP). Negative → back up.
        • gripper ∈ [-1, 1]  — thresholded at 0.0 (>0 → on)

    Observation space (53 dims):
        eef_pos(3) + obj_pos(3) + relative_obj_pos(3) + obj_orn(4)
        + relative_obj_vel(3) + eef_vel(3) + prev_eef_vel(3)
        + target_pos(3) + gripper(1) + target_pos_human(3)
        + human_pos_history(5×3=15) + human_vel(3) + prev_action(4) + human_obj_start(2)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    # ── robot constants ──────────────────────────────────────────────────── #
    ARM_JOINTS    = [0, 1, 2, 3, 4, 5]
    EEF_LINK      = 6
    TOOL_TIP_LINK = 7
    tool_tip_height = 0.061

    X_LIM = (0,      0.440)
    Y_LIM = (-0.440, 0.440)
    Z_LIM = (0.045,  0.683)

    MAX_STEPS   = 200
    MAX_SPEED   = 0.4        # m/s
    action_repeat = 10       # physics steps per env step (24 Hz control)

    grasp_dist    = 0.0252
    pick_height   = 0.045
    cruise_height = 0.06
    object_height = 0.025
    margin_error  = 0.003

    TRAJ_POINT_SPACING = 0.0001   #0.0001 metres between consecutive waypoints
    MAX_TRAJ_STEP      = 100       #100  maximum waypoint-steps per env step
    PICK_DWELL_STEPS   = 6       # trajectory points held at pick_height

    _grasp_constraint = None

    # ─────────────────────────────────────────────────────────────────────── #

    def __init__(self, render_mode=None, hz=240, eval_mode=False):
        super().__init__()
        self.render_mode  = render_mode
        self.hz           = hz
        self._physics_client = None
        self.eval_mode    = eval_mode
        self.teleport_on_success = True
        self.train_pick_up       = False

        # ── motion-capture data ─────────────────────────────────────────── #
        self.data = [None] * 4
        #self.data[0] = self.load_human_movement("mot_cap_data/real1.csv")
        #self.data[1] = self.load_human_movement("mot_cap_data/real2.csv")
        self.data[2] = self.load_human_movement("mot_cap_data/real3.csv")
        #self.data[3] = self.load_human_movement("mot_cap_data/normal3.csv")
        self.data[0] = self.load_human_movement("mot_cap_data/newnormal1.csv")
        self.data[1] = self.load_human_movement("mot_cap_data/newnormal2.csv")
        #self.data[2] = self.load_human_movement("mot_cap_data/normal2.csv")
        #self.data[3] = self.load_human_movement("mot_cap_data/normal3.csv")

        self.human_pos         = [[0, 0, 0]]
        self.prev_human_pos    = np.zeros(3, dtype=np.float32)
        self.human_pos_history = np.zeros((5, 3), dtype=np.float32)

        # ── spaces ──────────────────────────────────────────────────────── #
        obs_low  = np.full(53, -np.inf, dtype=np.float32)
        obs_high = np.full(53,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ── internal state ───────────────────────────────────────────────── #
        self._step_count      = 0
        self._gripper_active  = False
        self.in_pick_up       = False
        self._grasped         = False
        self._target_pos      = None
        self._object_id       = None
        self.human_reached_target = -1
        self.reached_target       = -1
        self.prev_action          = np.zeros(4, dtype=np.float32)
        self.hr_dist_ratio        = 1.0

        # ── trajectory state ─────────────────────────────────────────────── #
        self._trajectory   = None   # np.ndarray (N_total, 3)
        self._traj_index   = 0      # current pointer into _trajectory
        self._traj_splits  = []     # last index of each segment (4 entries)

    # ─────────────────────────────────────────────────────────────────────── #
    # Trajectory helpers
    # ─────────────────────────────────────────────────────────────────────── #

    def _build_trajectory(self, eef_pos: np.ndarray):
        """
        if self.render_mode == "real":
            try:
                import real_robot_control_mocap as rrcm
                obj_pose = rrcm.get_obj1_pose()
                if obj_pose is not None:
                    rx = obj_pose["position"]["x"]
                    ry = obj_pose["position"]["y"]
                    rz = obj_pose["position"]["z"]
                    
                    p.resetBasePositionAndOrientation(
                        self._object_id, [rx, ry, rz], [0, 0, 0, 1],
                        physicsClientId=self._physics_client
                    )
            except Exception as e:
                pass
            """

        obj_pos, _ = p.getBasePositionAndOrientation(
            self._object_id, physicsClientId=self._physics_client
        )
        #print("obj_pos trajectory sim ", obj_pos)
        eef_pos   = eef_pos.astype(np.float32)
        place_pos = self._target_pos.astype(np.float32)

        above_obj = np.array([obj_pos[0],   obj_pos[1],   self.cruise_height], dtype=np.float32)
        pick_pos  = np.array([obj_pos[0],   obj_pos[1],   self.pick_height],   dtype=np.float32)
        above_tgt = np.array([place_pos[0], place_pos[1], self.cruise_height], dtype=np.float32)

        seg0  = min_jerk_traj(eef_pos,   above_obj,  self.TRAJ_POINT_SPACING)
        seg1  = min_jerk_traj(above_obj, pick_pos,   self.TRAJ_POINT_SPACING)
        dwell = np.tile(pick_pos, (self.PICK_DWELL_STEPS, 1))
        seg2  = min_jerk_traj(pick_pos,  above_tgt,  self.TRAJ_POINT_SPACING)
        seg3  = min_jerk_traj(above_tgt, place_pos,  self.TRAJ_POINT_SPACING)

        self._trajectory = np.concatenate([
            seg0,
            seg1[1:],
            dwell,
            seg2[1:],
            seg3[1:],
        ], axis=0)

        n0 = len(seg0) - 1
        n1 = n0 + len(seg1) - 1 + self.PICK_DWELL_STEPS
        n2 = n1 + len(seg2) - 1
        n3 = n2 + len(seg3) - 1
        self._traj_splits = [n0, n1, n2, n3]
        self._traj_index  = 0

    def _current_segment(self) -> int:
        for seg_id, split in enumerate(self._traj_splits):
            if self._traj_index <= split:
                return seg_id
        return 3

    def _get_eef_pos(self) -> np.ndarray:
        _, _, _, _, eef_joint_pos, _, *_ = p.getLinkState(
            self._robot_id, self.EEF_LINK,
            physicsClientId=self._physics_client
        )
        return np.array(eef_joint_pos) - np.array([0, 0, self.tool_tip_height])

    def _move_to_waypoint(self, waypoint: np.ndarray, eef_orn):
        target = np.clip(waypoint,
                         [self.X_LIM[0], self.Y_LIM[0], self.Z_LIM[0]],
                         [self.X_LIM[1], self.Y_LIM[1], self.Z_LIM[1]])
        joint_poses = p.calculateInverseKinematics(
            self._robot_id, self.TOOL_TIP_LINK,
            target.tolist(), [1, 0, 0, 0],
            physicsClientId=self._physics_client
        )
        p.setJointMotorControlArray(
            self._robot_id,
            self.ARM_JOINTS,
            p.POSITION_CONTROL,
            targetPositions=joint_poses[:6],
            physicsClientId=self._physics_client
        )
        return joint_poses

    # ─────────────────────────────────────────────────────────────────────── #
    # Simulation helpers
    # ─────────────────────────────────────────────────────────────────────── #

    def _connect(self):
        if self._physics_client is not None:
            return
        mode = p.GUI if self.render_mode in ("human", "real") else p.DIRECT
        self._physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._physics_client)
        if self.render_mode not in ("human", "real"):
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self._physics_client)
        else:
            p.resetDebugVisualizerCamera(
                cameraDistance=1, cameraYaw=90, cameraPitch=-30,
                cameraTargetPosition=[0.15, 0, 0.1]
            )

    def load_human_movement(self, file_path):
        """
        Dynamically finds and maps the tracked columns for RigidBody 002.
        """

        # 1. Read first 7 lines to parse headers dynamically from top metadata rows
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header_rows = [next(reader) for _ in range(7)]

        type_row = header_rows[2]
        name_row = header_rows[3]
        prop_row = header_rows[5]
        comp_row = header_rows[6]

        # Map components to index for 'RigidBody 002'
        rb_rot = {comp_row[idx]: idx for idx in range(len(name_row)) if idx < len(type_row) and name_row[idx] == 'RigidBody 002' and type_row[idx] == 'Rigid Body' and prop_row[idx] == 'Rotation'}
        rb_pos = {comp_row[idx]: idx for idx in range(len(name_row)) if idx < len(type_row) and name_row[idx] == 'RigidBody 002' and type_row[idx] == 'Rigid Body' and prop_row[idx] == 'Position'}

        # Construct final ordered columns list
        cols = [
            0, 1,                        # Frame, Time
            rb_rot.get('X'), rb_rot.get('Y'), rb_rot.get('Z'), rb_rot.get('W'),  # qx, qy, qz, qw
            rb_pos.get('X'), rb_pos.get('Y'), rb_pos.get('Z')                   # x, y, z
        ]
        names = ['frame', 'time', 'qx', 'qy', 'qz', 'qw', 'x', 'y', 'z']

        # Determine structural lines to skip for data phase
        skip_map = {
            "mot_cap_data/constant-slow.csv":       187,
            "mot_cap_data/constant-medium_002.csv": 620,
            "mot_cap_data/natural-medium_003.csv":  340,
            "mot_cap_data/natural-slow_002.csv":    150,
            "mot_cap_data/newnormal1.csv":          236,
        }
        skiprows = skip_map.get(file_path, 7)
        
        # 2. Parse using Pandas with specific selected dynamic columns
        df = pd.read_csv(file_path, skiprows=skiprows, header=None, usecols=cols)

        # Realignment mappings because pandas processes integer list in sorted index layout
        col_to_name = {c: n for c, n in zip(cols, names)}
        sorted_cols = sorted(cols)
        df.columns = [col_to_name[c] for c in sorted_cols]
        
        # Ensure ordered layout array structure returned
        df = df[names]
        return df

    def step_human_movement(self, data, i):
        human_pos = [data.iloc[i]['x'], data.iloc[i]['y'], data.iloc[i]['z']]
        T_3d = FrameTransform.from_3d(
            roll=90, pitch=0, yaw=90,
            tx=0.97 - 0.059, ty=-0.337, tz=-0.8 + 0.033,
        )
        human_trans_pos = T_3d.transform(np.array(human_pos))
        #human_trans_pos[0][1] *= self.rand_direction

        p.resetBasePositionAndOrientation(self.human_id,       human_pos,                    [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.human_trans_id, human_trans_pos[0].tolist(),  [0, 0, 0, 1])
        return human_trans_pos

    def _load_scene(self):
        p.resetSimulation(physicsClientId=self._physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0,
                                   physicsClientId=self._physics_client)

        p.loadURDF("table/table.urdf",
                   basePosition=[0.5, 0, -0.65 + 0.025],
                   baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
                   physicsClientId=self._physics_client)

        self._robot_id = p.loadURDF(
            "models/lite6_with_vacuum.urdf", [0, 0, 0],
            useFixedBase=True, physicsClientId=self._physics_client
        )

        self._target_pos_human = np.array([0.64, -0.26 * self.rand_direction, 0.04])
        self.human_obj_start = np.array([0.64, 0.26 * self.rand_direction])
        self.hr_dist_ratio = 2.0

        if self.render_mode == "real":
            obj_x_random    = 0.25 + 0.025
            target_x_random = 0.25
            obj_y           = 0.07 + 0.025
            target_y        = -0.14
        elif not self.train_pick_up:
            target_x_random = 0.25
            obj_x_random    = target_x_random
            random_adjustment = np.random.uniform(-0.05, 0.05)
            obj_noise = np.random.uniform(-0.02, 0.02)
            obj_y    = (-0.11 + random_adjustment + obj_noise) * self.rand_direction
            target_y =  (0.11 + random_adjustment) * self.rand_direction
            self.hr_dist_ratio =  np.linalg.norm(self.human_obj_start[1] - self._target_pos_human[1]) / np.linalg.norm(obj_y - target_y)
            #print("self.hr_dist_ratio ",self.hr_dist_ratio)

            if self.eval_mode:
                self.obj_pos_start_array = [np.array([0.25, 0.08, 0.06]) , np.array([0.25, 0.14, 0.06]),
                                      np.array([0.30, 0.08, 0.06]), np.array([0.30, 0.14, 0.06]) ]
                
                self.target_array = [np.array([0.25, -0.14, 0.06]) , np.array([0.25, -0.10, 0.06]),
                                      np.array([0.30, -0.14, 0.06]), np.array([0.30, -0.10, 0.06]) ]
            #print("self.hr_dist_ratio ",self.hr_dist_ratio)
        else:
            obj_x_random    = np.random.uniform(0.18, 0.3)
            target_x_random = np.random.uniform(0.18, 0.3)
            obj_y    = np.random.uniform(-0.25, 0.25)
            target_y = np.random.uniform(-0.25, 0.25)

        self._obj_start = np.array([obj_x_random, obj_y, 0.08])
        self._object_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=self._obj_start.tolist(),
            physicsClientId=self._physics_client
        )

        self._target_pos = np.array([target_x_random, target_y, 0.06])
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                  rgbaColor=[1, 0, 0, 0.6],
                                  physicsClientId=self._physics_client)
        self.robot_target_id = p.createMultiBody(baseVisualShapeIndex=vis,
                          basePosition=self._target_pos.tolist(),
                          physicsClientId=self._physics_client)

        
        vis_human = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                        rgbaColor=[0, 1, 0, 0.6],
                                        physicsClientId=self._physics_client)
        p.createMultiBody(baseVisualShapeIndex=vis_human,
                          basePosition=self._target_pos_human.tolist(),
                          physicsClientId=self._physics_client)

        target_visual_id  = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        target_visual2_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
        self.human_id       = p.createMultiBody(baseVisualShapeIndex=target_visual_id,  basePosition=[0.5, 0, 0.5])
        self.human_trans_id = p.createMultiBody(baseVisualShapeIndex=target_visual2_id, basePosition=[0.5, 0, 0.5])

    # ─────────────────────────────────────────────────────────────────────── #
    # Gym API
    # ─────────────────────────────────────────────────────────────────────── #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self.rand_direction = random.choice([-1, 1])
        self.rand_direction = -1
        self._load_scene()

        self._step_count         = 0
        self.prev_eef_vel        = np.zeros(3)
        self.prev_human_pos      = np.zeros(3, dtype=np.float32)
        self.human_pos_history   = np.zeros((5, 3), dtype=np.float32)
        self.prev_action         = np.zeros(4, dtype=np.float32)
        self._gripper_active     = False
        self._grasped            = False
        self._grasp_bonus_given  = False
        self._lift_bonus_given   = False
        self.in_pick_up          = False
        self.reached_pick_height = False
        self.reached_target      = -1
        self.human_reached_target = -1
        self.n_goal_complete      = 0
        self.enable_human_reach   = False
        self.human_index          = random.randint(0, 2)
        self.human_speed          = random.randint(3, 5)
        self.real_reached         = False
        self.obj_index = 0
        
        if self.eval_mode:
            self.human_index = 1
            self.human_speed = 4

        for _ in range(10):
            p.stepSimulation(physicsClientId=self._physics_client)
        eef_pos = self._get_eef_pos()
        self._build_trajectory(eef_pos)

        obs  = self._get_obs()
        info = {}
        return obs, info

    def _reset_object(self):
        if not self.render_mode  == "real":

            if self.render_mode == "human" and self.n_goal_complete <= 3:
                obj_reset = self.obj_pos_start_array[self.n_goal_complete]
                self._target_pos = self.target_array[self.n_goal_complete]
                    
            else:
                target_x_random = np.random.uniform(0.25, 0.30)
                obj_x_random    = target_x_random
                random_adjustment = np.random.uniform(-0.06, 0.06)
                obj_y    = (-0.11 - random_adjustment) * self.rand_direction
                target_y =  (0.11 + random_adjustment) * self.rand_direction
                
                obj_reset = np.array([obj_x_random, obj_y, 0.08])
                self._target_pos = np.array([target_x_random, target_y, 0.06])
                self.hr_dist_ratio =  np.linalg.norm(self.human_obj_start[1] - self._target_pos_human[1]) / np.linalg.norm(obj_y - target_y)

            #change target pos
            p.resetBasePositionAndOrientation(self.robot_target_id,
                    self._target_pos.tolist(), [0, 0, 0, 1],
                    physicsClientId=self._physics_client
                )
            #change obj pos
            p.resetBasePositionAndOrientation(
                self._object_id,
                obj_reset.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self._physics_client
            )
            if self._grasped and self._grasp_constraint is not None:
                p.removeConstraint(self._grasp_constraint,
                                physicsClientId=self._physics_client)
                self._grasped        = False
                self._gripper_active = False

        self._grasp_bonus_given   = False
        self.reached_target       = -1
        self.human_reached_target = -1

        eef_pos = self._get_eef_pos()
        self._build_trajectory(eef_pos)

    def set_curriculum_stage(self, stage: int):
        self.teleport_on_success = (stage >= 1)
        if self.teleport_on_success:
            print(f"[Env] Curriculum stage {stage} — teleport on success enabled")

    # ─────────────────────────────────────────────────────────────────────── #
    # Step
    # ─────────────────────────────────────────────────────────────────────── #

    def step(self, action):
        speed_raw = float(np.clip(action[0], -1.0, 1.0))
        speed_cmd = (speed_raw + 1.0) * 0.5
        grip_cmd  = float(np.clip(action[1], -1.0, 1.0))

        #speed_cmd = 0.7
        #grip_cmd = 1
        #print("speed_cmd ", speed_cmd)
        #print("grip_cmd ", grip_cmd)

        traj_delta = int(np.round(speed_cmd * self.MAX_TRAJ_STEP))
        self._traj_index = int(np.clip(
            self._traj_index + traj_delta,
            0, len(self._trajectory) - 1
        ))
        desired_waypoint = self._trajectory[self._traj_index]
        current_segment  = self._current_segment()

        self.action_delta = np.abs(
            np.array([speed_cmd, grip_cmd, 0.0, 0.0]) - self.prev_action
        )
        self.prev_action = np.array([speed_cmd, grip_cmd, 0.0, 0.0], dtype=np.float32)

        joint_poses = None
        for _ in range(self.action_repeat):
            obj_pos, obj_orn = p.getBasePositionAndOrientation(
                self._object_id, physicsClientId=self._physics_client
            )
            obj_pos = np.array(obj_pos)

            _, _, _, _, eef_joint_pos, eef_orn, *_ = p.getLinkState(
                self._robot_id, self.EEF_LINK,
                physicsClientId=self._physics_client
            )
            eef_pos = np.array(eef_joint_pos) - np.array([0, 0, self.tool_tip_height])

            dist_to_obj     = np.linalg.norm(eef_pos - obj_pos)
            dist_to_obj_x_y = np.linalg.norm(eef_pos[:2] - obj_pos[:2])
            dist_to_obj_z   = np.linalg.norm(eef_pos[2]  - obj_pos[2])

            joint_poses = self._move_to_waypoint(desired_waypoint, eef_orn)

            self._gripper_active = grip_cmd > 0.0

            if (self._gripper_active
                    and dist_to_obj_x_y <= 0.020
                    and dist_to_obj_z   <= self.grasp_dist + self.margin_error
                    and not self._grasped):
                self._grasp_constraint = p.createConstraint(
                    self._robot_id, self.TOOL_TIP_LINK,
                    self._object_id, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0], [0, 0, 0.025], [0, 0, 0],
                    parentFrameOrientation=eef_orn,
                    childFrameOrientation=obj_orn,
                    physicsClientId=self._physics_client
                )
                self._grasped = True

            if not self._gripper_active and self._grasped:
                p.removeConstraint(self._grasp_constraint,
                                   physicsClientId=self._physics_client)
                self._grasped = False

            self.human_pos = self.step_human_movement(
                self.data[self.human_index],
                self._step_count * self.human_speed
            )
            p.stepSimulation(physicsClientId=self._physics_client)
            if self.render_mode == "human":
                time.sleep(1.0 / self.hz)

        obs = self._get_obs()
        reward, terminated = self._compute_reward(obs)

        self._step_count += 1
        if self.eval_mode or self.render_mode == "real":
            truncated = self._step_count >= self.MAX_STEPS * 1.5 * 1.0/(self.human_speed / 5.0)
        elif self.teleport_on_success:
            truncated = self._step_count >= self.MAX_STEPS * 1.2 * 1.0/(self.human_speed / 5.0)
        else:
            truncated = self._step_count >= self.MAX_STEPS

        info = {
            "reached_target":       self.reached_target,
            "human_reached_target": self.human_reached_target,
            "real_reached_target":  self.real_reached,
            "dist_eef_to_obj":      dist_to_obj,
            "grasped":              self._grasped,
            "joint_poses":          joint_poses,
            "is_success":           terminated,
            "traj_index":           self._traj_index,
            "traj_length":          len(self._trajectory),
            "traj_segment":         current_segment,
            "traj_splits":          self._traj_splits,
            "n_goal_complete":      self.obj_index,
        }
        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────── #
    # Observation
    # ─────────────────────────────────────────────────────────────────────── #

    def _get_obs(self):
        _, _, _, _, eef_joint_pos, eef_orn, eef_vel, _ = p.getLinkState(
            self._robot_id, self.EEF_LINK, computeLinkVelocity=1,
            physicsClientId=self._physics_client
        )
        eef_vel = np.array(eef_vel)
        eef_pos = np.array(eef_joint_pos) - np.array([0, 0, self.tool_tip_height])

        obj_pos, obj_orn = p.getBasePositionAndOrientation(
            self._object_id, physicsClientId=self._physics_client
        )
        obj_pos = np.array(obj_pos)
        obj_pos[2] += self.object_height
        obj_orn = np.array(obj_orn)

        relative_obj_pos = obj_pos - eef_pos

        obj_vel, _ = p.getBaseVelocity(self._object_id)
        obj_vel = np.array(obj_vel)
        relative_obj_vel = obj_vel - eef_vel

        gripper_state = [1.0 if self._gripper_active else 0.0]

        human_pos = np.array(self.human_pos[0])
        self.human_pos_history = np.roll(self.human_pos_history, shift=-1, axis=0)
        self.human_pos_history[-1] = human_pos

        dt = self.action_repeat / self.hz
        human_vel = (human_pos - self.prev_human_pos) / dt
        self.prev_human_pos = human_pos.copy()

        obs = np.concatenate([
            eef_pos,                           # 3   0:3
            obj_pos,                           # 3   3:6
            relative_obj_pos,                  # 3   6:9
            obj_orn,                           # 4   9:13
            relative_obj_vel,                  # 3   13:16
            eef_vel,                           # 3   16:19
            self.prev_eef_vel,                 # 3   19:22
            self._target_pos,                  # 3   22:25
            gripper_state,                     # 1   25
            self._target_pos_human,            # 3   26:29
            self.human_pos_history.flatten(),  # 15  29:44
            human_vel,                         # 3   44:47
            self.prev_action,                  # 4   47:51
            self.human_obj_start,              # 2   51:53
        ]).astype(np.float32)

        self.prev_eef_vel = eef_vel
        return obs

    # ─────────────────────────────────────────────────────────────────────── #
    # Reward
    # ─────────────────────────────────────────────────────────────────────── #

    def _compute_reward(self, obs):
        terminated = False

        obj_pos      = obs[3:6] - self.object_height
        eef_pos      = obs[:3]
        eef_vel      = obs[16:19]
        prev_eef_vel = obs[19:22]
        human_vel    = obs[44:47]

        obj_pos = np.array(obj_pos)

        dist_eef_obj     = np.linalg.norm(eef_pos[:2] - obj_pos[:2])
        dist_eef_obj_z   = np.linalg.norm(eef_pos[2]  - obj_pos[2])
        dist_obj_tgt_x_y = np.linalg.norm(obj_pos[:2] - self._target_pos[:2])
        dist_obj_tgt_x   = np.linalg.norm(obj_pos[0]  - self._target_pos[0])
        dist_obj_tgt_z   = np.linalg.norm(obj_pos[2]  - self._target_pos[2])
        dist_eff_tgt_y   = np.linalg.norm(eef_pos[1]  - self._target_pos[1])

        reward = 0.0

        smoothness_penalty = np.sum(np.square(self.action_delta[0])) * 0.05
        reward -= smoothness_penalty

        dist_human_target_xy = np.linalg.norm(
            self.human_pos[0][:2] - self._target_pos_human[:2])
        dist_human_target_y  = np.linalg.norm(
            self.human_pos[0][1]  - self._target_pos_human[1])

        #print("smoothness_penalty ",smoothness_penalty)
        #print("dist ",np.linalg.norm(dist_human_target_y / self.hr_dist_ratio - dist_eff_tgt_y) * 1.5)

        if self.teleport_on_success or self.eval_mode:
            reward -= np.linalg.norm(
                dist_human_target_y / self.hr_dist_ratio - dist_eff_tgt_y) * 1.5
            reward -= np.abs(
                np.linalg.norm(human_vel[:2] / self.hr_dist_ratio)
                - np.linalg.norm(eef_vel[:2]))
        else:
            reward -= dist_eff_tgt_y

        if self._grasped and not self._grasp_bonus_given:
            reward += 5.0
            self._grasp_bonus_given = True

        if self.human_pos[0][1] * self.rand_direction > 0:
            self.enable_human_reach = True
            self.human_reached_target = -1

        if dist_obj_tgt_x_y < 0.04 and self.reached_target < 0 and not self._grasped:
            self.reached_target = self._step_count
            self.obj_index = self.n_goal_complete
        #print("self.human_pos ",self.human_pos)
        if dist_human_target_y < 0.06 and self.human_reached_target < 0 :# and self.enable_human_reach:
            #print("self. self._step_count[:2] ", self._step_count)
            self.human_reached_target = self._step_count
            #print("self.human_reached_target ",self.human_reached_target, self.human_speed)

        if self.real_reached:# and dist_obj_tgt_x_y > 0.05:
            #print("reset object")
            #self._reset_object()
            self.real_reached = False
            """
            self.n_goal_complete     += 1
            if self.n_goal_complete >= 2:
                    terminated = True
                    """

        #print("self.reached_target ",self.reached_target, self.render_mode)
        if self.reached_target >= 0 and (self.human_reached_target > 0 or self.render_mode == "real"):
            reward += 10.0 #- abs(self.reached_target - self.human_reached_target) * 1.0

            if self.render_mode == "real":
                #print("one")
                #terminated = True
                self.real_reached = True
            elif self.eval_mode or self.teleport_on_success :#elif
                
                self.n_goal_complete     += 1
                if self.n_goal_complete >= 4:
                    terminated = True
                if self.render_mode == "human":
                    print(f"robot t={self.reached_target} | "
                          f"human t={self.human_reached_target}")
                    
                self._reset_object()
                self.reached_target       = -1
                self.human_reached_target = -1
                self.enable_human_reach   = False
            else:
                terminated = True

        return float(reward), terminated

    def render(self):
        if self.render_mode == "human":
            w, h = 640, 480
            view_mat = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.5, 0, 0.2],
                distance=0.5, yaw=45, pitch=-30, roll=0,
                upAxisIndex=2, physicsClientId=self._physics_client
            )
            proj_mat = p.computeProjectionMatrixFOV(
                fov=60, aspect=w / h, nearVal=0.1, farVal=5.0,
                physicsClientId=self._physics_client
            )
            _, _, rgb, _, _ = p.getCameraImage(
                w, h, view_mat, proj_mat,
                physicsClientId=self._physics_client
            )
            return np.array(rgb, dtype=np.uint8)[:, :, :3]

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import matplotlib.pyplot as plt
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CSV Logging  (mirrors real_robot_control_mocap.py)
# ─────────────────────────────────────────────────────────────────────────────

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
        path = os.path.join("eval_results", f"eval_trajectory_{ts}.csv")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    _csv_file   = open(path, "w", newline="")
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
    print(f"CSV logger opened: {path}")
    return path


def log_state(timestep,
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
        timestep,
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
        print("CSV logger closed.")


def save_eval_trajectory(trajectory: list, filename: str = "eval_trajectory.csv"):
    import csv
    os.makedirs("eval_results", exist_ok=True)
    filepath = os.path.join("eval_results", filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "step", "x", "y", "z", "grasped", "reward"])
        writer.writeheader()
        writer.writerows(trajectory)
    print(f"Trajectory saved → {filepath}")


def all_plot_rewards():
    #data = np.load("cluster_ppo_logs/evaluations.npz")
    data = np.load("ppo_logs/evaluations.npz")
    timesteps    = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    std_rewards  = data["results"].std(axis=1)
    has_success  = "successes" in data

    fig, axes = plt.subplots(
        2 if has_success else 1, 1,
        figsize=(10, 7 if has_success else 4), sharex=True
    )
    if not has_success:
        axes = [axes]

    axes[0].plot(timesteps, mean_rewards, label="mean reward")
    axes[0].fill_between(timesteps,
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.2, label="±1 std")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Evaluation Reward over Training")
    axes[0].legend()

    if has_success:
        success_rate = data["successes"].mean(axis=1)
        axes[1].plot(timesteps, success_rate, color="green", label="success rate")
        axes[1].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        axes[1].set_ylabel("Success Rate")
        axes[1].set_title("Evaluation Success Rate over Training")
        axes[1].legend()
    else:
        print("No 'successes' key in evaluations.npz.")

    axes[-1].set_xlabel("Timesteps")
    plt.tight_layout()
    plt.show()

def plot_rewards():
    #data = np.load("cluster_ppo_logs/evaluations.npz")
    data = np.load("ppo_logs/evaluations.npz")
    timesteps    = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    std_rewards  = data["results"].std(axis=1)

    mask = timesteps <= 2_400_000
    timesteps    = timesteps[mask]
    mean_rewards = mean_rewards[mask]
    std_rewards  = std_rewards[mask]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(timesteps, mean_rewards, label="mean reward")
    ax.fill_between(timesteps,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.2, label="±1 std")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Timesteps")
    ax.set_title("Evaluation Reward over Training")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #model_path   = "cluster_ppo_lite6_pick_place_ppo_final"
    #vecnorm_path = "cluster_ppo_vec_normalize.pkl"
    model_path   = "ppo_lite6_pick_place_ppo_final"
    vecnorm_path = "ppo_vec_normalize.pkl"
    #model_path   = "good_models/sync10_premade3/ppo_lite6_pick_place_ppo_final"
    #vecnorm_path = "good_models/sync10_premade3/ppo_vec_normalize.pkl"

    env = DummyVecEnv([lambda: Lite6PickPlaceEnv(render_mode="human", eval_mode=True)])
    env = VecNormalize.load(vecnorm_path, env)
    env.training    = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    obs   = env.reset()

    plot_rewards()

    # ── CSV logger ────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_csv_logger(os.path.join("experiment_data/sim_results", f"eval_trajectory_{ts}.csv"))

    episode      = 0
    step_in_ep   = 0
    prev_eef_pos = np.zeros(3, dtype=np.float32)
    prev_eef_vel = np.zeros(3, dtype=np.float32)

    try:
        for _ in range(500 * 4):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(action)

            # ── Extract state for logging ─────────────────────────────────────
            raw_obs = obs[0]                          # unwrap VecEnv batch dim

            eef_pos  = raw_obs[0:3].copy()
            obj_pos  = raw_obs[3:6].copy()
            human_pos = raw_obs[29:32].copy()         # latest entry in history
            human_vel = raw_obs[44:47].copy()
            prev_eef_vel = raw_obs[16:19].copy()

            # Approximate eef_vel from position delta (1/24 s step)
            eef_vel      = (eef_pos - prev_eef_pos) / (1.0 / 24.0)
            prev_eef_pos = eef_pos.copy()

            step_info = info[0] if isinstance(info, (list, tuple)) else info

            log_state( timestep = step_in_ep * (1.0/24.0),    
            eef_pos       = prev_eef_pos,
            eef_vel       = prev_eef_vel,
            human_pos     = human_pos,
            human_vel     = human_vel,
            obj_pos       = obj_pos,
            active_obj_idx= info[0]["n_goal_complete"],
            )

            step_in_ep += 1

            if done[0]:
                obs          = env.reset()
                episode     += 1
                step_in_ep   = 0
                prev_eef_pos = np.zeros(3, dtype=np.float32)
                prev_eef_vel = np.zeros(3, dtype=np.float32)
                break

    finally:
        close_csv_logger()
        env.close()