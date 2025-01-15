from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np  

class TwsbrEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "simulation_fps": 60}

    def __init__(self,
                render_mode=None,
                action_type="continuous",
                wheels_controlled_together = False,
                roll_threshold_deg = 30.0,          # terminate after body tilt reaches roll_threshold deg (tilt)
                x_threshold = 10.0,                 # terminate after robot moves more than x_threshold [m]
                y_threshold = 10.0, ):              # terminate after robot moves more than y_threshold [m]
        
        #configs
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        assert action_type in ["binary", "discrete", "continuous"]
        self._action_type = action_type
        self._wheel_controlled_together = wheels_controlled_together
        self._render_height = parameters["render"]["render_height"]        # For render_mode="rgb_array"
        self._render_width= parameters["render"]["render_width"]           # For render_mode="rgb_array"
        self._physics_client_id = -1
        self.roll_threshold_rad = roll_threshold_deg / 180.0 * np.pi
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.this_file_dir_path = os.path.abspath(os.path.dirname(__file__))
        self.urdf_file_name = parameters["urdf_file_name"]

        # State := [roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot] 
        state_limit = np.array([np.pi,                      # roll: tilt angle constrained in range # pitch
                                np.pi,                      # pith
                                np.pi,                      # yaw
                                np.finfo(np.float32).max,   # omega_x
                                np.finfo(np.float32).max,   # omgea_y
                                np.finfo(np.float32).max,   # omega_z
                                self.x_threshold,           # x coordinate constrained in range
                                self.y_threshold,           # y coordinate constrained in range.
                                np.finfo(np.float32).max,   # z
                                np.finfo(np.float32).max,   # x_dot
                                np.finfo(np.float32).max,   # y_dot
                                np.finfo(np.float32).max,   # z_dot
                                ])
        self.observation_space = gym.spaces. Box(low = -state_limit,
                                                 high = state_limit)

        #Action
        if self._action_type == "binary":
            # 2 options [forward, backward], both apply same torque magnitude to 2 wheels
            self.binary_action_torque_magnitude = parameters ["binary_action"]["torque_magnitude"]
            if self._wheel_controlled_together:
                #dim_action = 1: controls both wheels 
                self.action_space = gym.spaces.Discrete(2) # 2 actions will be 0: -binary_action_torque_magnitude, 1: +binary_action_torque_magnitude
            else:
                #dim_action = 2: independent control 
                self.action_space = gym.spaces.Discrete(2*2) # 2 wheels each has 2 action options

        elif self._action_type == "discrete":
            # N options to provide finer control, list of motor torques in ascending order 
            self.discrete_action_torque_magnitudes = parameters["discrete_action"] ["torque_magnitudes"]
            if self._wheel_controlled_together:
                #dim_action = 1: controls both wheels
                self.action_space = gym.spaces.Discrete(len(self.discrete_action_torque_magnitudes))
            else: 
                #dim_action = 2: independent control
                self.action_space = gym.spaces.Discrete(2*len(self.discrete_action_torque_magnitudes)) # as a simplification, assume one wheel control each step (2*N options) otherwise N**2 options 
        
        elif self._action_type == "continuous":
            # torque in range (-self.max_torque_magnitude, +self.max_torque_magnitude)
            self.continuous_action_max_torque_magnitude = parameters["continuous_action"]["max_torque_magnitude"]
            if self._wheel_controlled_together:
                #dim_action = 1: controls both wheels
                self.action_space= gym.spaces.Box(low=[-self.continuous_action_max_torque_magnitude],
                                                  high=[self.continuous_action_max_torque_magnitude])
            else:
                #dim_action = 2: independent control
                self.action_space = gym.spaces.Box (low=np.array([-self.continuous_action_max_torque_magnitude] *2),
                                                    high=np.array([self.continuous_action_max_torque_magnitude] *2))

        return
        
    def reset(self, 
              seed=None,
              options=None,
              start_position=None,
              start_tilt_deg = None,
              start_yaw_deg = 0.0,
              ):
        super().reset(seed=seed) # after setting seed, use self.np_random as generator
        # setup PyBullet client, if not done yet
        if self._physics_client_id < 0:
            # create BulletClient
            if self.render_mode == "human":
               self._bullet_client = bullet_client.BulletClient (connection_mode=pybullet.GUI)
            else:
                self._bullet_client = bullet_client.BulletClient()
                
            self._physics_client_id = self._bullet_client._client
            self._bullet_client.resetSimulation()
            self._bullet_client.setGravity (0, 0, -9.8)
            self._bullet_client.setTimeStep(1.0/self.metadata["simulation_fps"])
            
            #load ground plane
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.ground_plane = self._bullet_client.loadURDF ("plane.urdf") # need this to find URDF for ground plane
                
            # load robot
            self.twsbr = self._bullet_client.loadURDF(os.path.join(self.this_file_dir_path, self.urdf_file_name))

        #initialize a new episode
        self._bullet_client.removeBody(self.twsbr) # remove old one from previous episode
        if start_tilt_deg is None:
            init_roll = self.np_random.uniform(low=-0.2, high=0.2, size=1)
        else:
            init_roll = start_tilt_deg / 180.0 * np.pi

        if start_yaw_deg is None:
            init_yaw = self.np_random.uniform (low=-np.pi, high=np.pi, size=1)
        else:
            init_yaw = start_yaw_deg / 180.0 * np.pi

        start_orientation = pybullet.getQuaternionFromEuler([init_roll, 0, init_yaw])
        if start_position is None:
           start_position= [0.0, 0.0, 0.001]

        self.twsbr = self._bullet_client.loadURDF(os.path.join(self.this_file_dir_path, self.urdf_file_name),
                                                     basePosition = start_position,
                                                     baseOrientation = start_orientation)

        # turn OFF default joint velocity control, torque control applied in step()
        self._bullet_client.setJointMotorControl2(self.twsbr, 0, pybullet. VELOCITY_CONTROL, force=0)
        self._bullet_client.setJointMotorControl2(self.twsbr, 1, pybullet. VELOCITY_CONTROL, force=0)
        self.step_counter = 0
        observation = self._get_obs()
        info= self._get_info()
        return observation, info
    
    def step(self, action):
        if self._action_type == "binary":
            if self._wheel_controlled_together:
                torque1 = self.binary_action_torque_magnitude if action == 1 else -self.binary_action_torque_magnitude
                torque2 = torque1
            else:
                if action == 0:
                    torque1 = self.binary_action_torque_magnitude
                    torque2 = 0.0
                elif action == 1:
                    torque1 = self.binary_action_torque_magnitude
                    torque2 = 0.0
                elif action == 2:
                    torque1 = 0.0
                    torque2 = self.binary_action_torque_magnitude
                elif action == 3:
                    torque1 = 0.0
                    torque2 = self.binary_action_torque_magnitude
                    
        elif self._action_type == "discrete":
            if self._wheel_controlled_together:
                torque1 = self.discrete_action_torque_magnitudes[action]
                torque2 = torque1
            else:
                # simplifying assumption at ONLY 1 wheel is controlled at each step, otherwise the combinatorial action has N**2 options 
                if action < len(self.discrete_action_torque_magnitudes):
                    torque1 = self.discrete_action_torque_magnitudes[action]
                    torque2 = 0.0
                else:
                    torque1 = 0.0
                    torque2 = self.discrete_action_torque_magnitudes[action - len(self.discrete_action_torque_magnitudes)]
        elif self._action_type == "continuous":
            if self._wheel_controlled_together:
                 torque1 = np.clip(action, self.continuous_action_max_torque_magnitude, self.continuous_action_max_torque_magnitude)
                 torque2 = torque1
            else: 
                torque1 = np.clip(action[0], self.continuous_action_max_torque_magnitude, self.continuous_action_max_torque_magnitude)
                torque2 = np.clip(action[1], -self.continuous_action_max_torque_magnitude, self.continuous_action_max_torque_magnitude)

        # apply torque
        self._bullet_client.setJointMotorControl2(body_UniqueId = self.twsbr,
                                                    jointIndex = 0,
                                                    controlMode = self._bullet_client.TORQUE_CONTROL, 
                                                    force = torque1)
        self._bullet_client.setJointMotorControl2(body_UniqueId = self.twsbr,
                                                    jointIndex = 1,
                                                    controlMode = self._bullet_client.TORQUE_CONTROL,
                                                    force = torque2)
        self._bullet_client.stepSimulation()

        observation = self._get_obs()
        roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = observation
        reward = self._get_reward()
        info = self._get_info()

        self.step_counter += 1
        if self.step_counter >= parameters ["truncation_steps"]:
            truncated = True
            info["is_success"] = True   # needed by SB3 logger
        else:
            truncated = False

        terminated = abs(roll) > self.roll_threshold_rad
        if terminated:
            info["is_success"] = False  # needed by SB3 logger

        return observation, reward, terminated, truncated, info
    
    
    def render(self):
        if self.render_mode== "rgb_array":
            # return a rgb_array of the current frame
            camera_dist = 1.2
            camera_pitch = 0.0
            camera_yaw = 160.0

        # get camera frame
        if (self._physics_client_id >= 0):
            view_matrix = self._bullet_client.computeViewMatrixFromYawPitchRoll (cameraTargetPosition= [0.0, 0.0, 0.5],
                                                                                 distance = camera_dist,
                                                                                 yaw = camera_yaw,
                                                                                 pitch = camera_pitch,
                                                                                 roll = 0.0,
                                                                                 upAxisIndex=2,
                                                                                 )
            projection_matrix = self._bullet_client.computeProjectionMatrixFOV (fov=60.0,
                                                                                aspect = float(self._render_width)/float(self._render_height),
                                                                                nearVal =  0.1,
                                                                                farVal = 100.0,
                                                                                )
            _, _, camera_frame, _, _ = self._bullet_client.getCameraImage(width = int(self._render_width),
                                                                          height= int(self._render_height),
                                                                          renderer =  self._bullet_client.ER_BULLET_HARDWARE_OPENGL,
                                                                          viewMatrix = view_matrix,
                                                                          projectionMatrix = projection_matrix,
                                                                          )
                                                                        
            rgb_array = np.array(camera_frame, dtype=np. uint8)
            rgb_array = np.reshape(rgb_array, (self._render_height, self._render_width, -1)) [:, :, :3]

        else:
            # if client NOT connected to PyBullet => blank image
            rgb_array = np.ones((int(self._render_width), int(self._render_height), 3), dtype=np.uint8) * 255

        return rgb_array

    def _get_reward(self):
        roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = self._get_obs()
       
        reward = 1.0                                                        # each step it maintain balance
        reward -= abs(pitch) * self._tilt_penalty_scale                     # penalty for deviation from up-straight position
        reward -= abs(pitch_dot) * self._tilt_speed_penalty_scale           # penalty for oscillation speed
        return reward

    def close(self):
        if self._physics_client_id >=0:
            self._bullet_client.disconnect()
        self._physics_client_id = -1
        print(f"Env Closed...")
        return