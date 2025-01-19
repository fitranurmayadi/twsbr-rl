from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2
import random
import math
import os
import sys
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

class TwsbrEnv(gym.Env):
    '''
        self-balancing mode = try to balance on the point
        waypoint mode = try to balance and go to target
        line-following mode = try to balance and follow the line
        self-driving mode = try to balance and go    to somewhere by the lidar
    
    Action : Discrete (42| 21 Left, 21 Right), Continuous (2 | Left Right | -255 to 255)
    State(Observation) : (24| prev_roll, prev_pitch, prev_yaw, prev_omega_x, prev_omega_y, prev_omega_z, prev_x, prev_y, prev_z, prev_x_dot, prev_y_dot, prev_z_dot, roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot)

    '''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    def __init__(self,
                render_mode=None,
                robot_angle_limit=45,
                action_type="continuous",
                max_velocity=255,
                truncation_steps=10000,
                debug_info = False
                ):

        super(TwsbrEnv, self).__init__()

        # Bottom camera settings
        self.projection_matrix = pybullet.computeProjectionMatrixFOV(fov=160.0, aspect=1.0, nearVal=0.0055, farVal=1)
        
        # Variabel global untuk menyimpan posisi terakhir garis
        self.prev_error_steer = 0  # Posisi terakhir yang terdeteksi
        self.line_last_position = None  # None, "left", atau "right"

        # Link indices
        self.CAMERA_IDX = 2
        self.CAMERA_TARGET_IDX = 3
        self.IMU_IDX = 4
        
        self.LEFT_WHEEL_JOINT_IDX = 0
        self.RIGHT_WHEEL_JOINT_IDX = 1

        # Default direction of the camera_up_vector
        self.camera_up_vector = np.array([0, -1, 0])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.robot_angle_limit = robot_angle_limit
        
        assert action_type in ["discrete", "continuous"]
        self._action_type = action_type
        
        self.max_velocity = max_velocity
        self.discrete_velocity = np.linspace(-self.max_velocity, self.max_velocity, num=21)

        self.truncation_steps = truncation_steps
        self.debug_info = debug_info

        # Observation and action space

        # x_dot (velocity), pitch(balance angle), yaw(line angle), motor_left_speed, motor_right_speed
        # +-25 m/s , +- 90 degree(+-pi rad), +-16 units, left wheel, and right wheel ~ in  +-255 rad/s ( +- 40.6 rps or +- 1200 rpm) 
        # add target speed as observation

        #self.state_limit = np.array([25.0, np.pi, 4.0, 255.0, 255.0, 25.0, np.pi, 4.0, 25.0, 25.0])
        self.state_limit = np.array([25.0, 25.0, 25.0, 25.0, np.pi, 4.0])
        
        self.observation_space = spaces.Box(low=-np.ones(self.state_limit.shape), high=np.ones(self.state_limit.shape)) 
        self.action_space = spaces.Discrete(len(self.discrete_velocity)*2) if action_type == "discrete" else spaces.Box(low=-1.0, high=1.0, shape=(2,)) # normalize version
        
        
        #self.obs_min = np.array([-25.0, -np.pi, -4.0, -25.0, -25.0, -25.0, -np.pi, -4.0, -25.0, -25.0])
        #self.obs_max = np.array([25.0, np.pi, 4.0, 25.0, 25.0, 25.0, np.pi, 4.0, 25.0, 25.0])
        
        self.obs_min = np.array([-25.0, -25.0, -25.0, -25.0, -np.pi, -4.0])
        self.obs_max = np.array([25.0, 25.0, 25.0, 25.0, np.pi, 4.0])
        
        self._physics_client_id = -1
        self.load_robot()
        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.load_robot()
        self.step_counter = 0
        self.prev_error_steer = 0  # Posisi terakhir yang terdeteksi
        self.line_last_position = None  # None, "left", atau "right"
        #self.previous_state = self._get_first_obs().astype(np.float32)
        return self._get_obs().astype(np.float32), self._get_info() 

    def load_robot(self):
        
        if self._physics_client_id < 0:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(pybullet.GUI, options="--width=1920 --height=1000")
                self.rendered_status = False
            else:
                self._bullet_client = bullet_client.BulletClient(pybullet.DIRECT)
            self._init_physics_client()
        else:
            self._bullet_client.removeBody(self.robot_id)
            #self._bullet_client.resetSimulation()
            self._init_physics_client()

    def _init_physics_client(self):
        
        self._physics_client_id = self._bullet_client._client
        # Set up simulation environment
        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, -9.8)
        self._bullet_client.setTimeStep(1.0 / self.render_fps)
        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        
        #pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane and robot
        
        self.texture = os.path.join(os.path.abspath(os.path.dirname(__file__)), "texture", "random_line_trace_ground.png")
        self.plane_id = self._bullet_client.loadURDF("plane.urdf", basePosition=[4.50, -0.62, 0.0], globalScaling=2.0)
        
        self.tex_uid =  self._bullet_client.loadTexture(self.texture)
        self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)
        
        #self.tex_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "texture", "line_trace_ground.png")
        #self.tex_uid = self._bullet_client.loadTexture(self.tex_file_name)
        #self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)
        
        self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")
      

        
        #DOMAIN RANDOMIZATION

        if self.render_mode == "human":
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/8, np.pi/8), random.uniform(-np.pi/8, np.pi/8)]) 
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, np.pi/4]) 
            self.target_speed = 0.1
            
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, 0])

        else:  #for training, give it more challenge         
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/8, np.pi/8), random.uniform(-np.pi, np.pi)])
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, 0]) 
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, 0]) 
        
            self.target_speed = round(np.random.uniform(0.05, 0.15), 2) # randomize target speed
            #self.plane_lateral_friction = np.random.uniform(0.7, 1.0)  # lateral friction
            #self.plane_spinning_friction = np.random.uniform(0.5, 1.0)  # lateral friction
            
            #self.robot_mass = np.random.uniform(0.500, 0.550)  # Random mass from 500 to 550 grams
            #self.robot_inertia = np.random.uniform(0.0001, 0.001, size=3)  # Random inertia [Ixx, Iyy, Izz]
            #self.robot_lateral_friction = np.random.uniform(0.7, 1.0)  # lateral friction
            #self.robot_spinning_friction = np.random.uniform(0.5, 1.0)  # lateral friction
            #self.robot_wheels_inertial = np.random.uniform(0.0, 0.001, size=3)  # Random inertia [Ixx, Iyy, Izz]

        self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name, basePosition=start_position, baseOrientation=start_orientation)

        self._bullet_client.setJointMotorControl2(self.robot_id, self.LEFT_WHEEL_JOINT_IDX, pybullet.VELOCITY_CONTROL, targetVelocity=0)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX, pybullet.VELOCITY_CONTROL, targetVelocity=0)

    def step(self, action):
        left_wheel_velocity, right_wheel_velocity = self._apply_action(action)

        self._bullet_client.setJointMotorControl2(self.robot_id, 0, pybullet.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity)
        self._bullet_client.setJointMotorControl2(self.robot_id, 1, pybullet.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity)
        self._bullet_client.stepSimulation()
        
        self.left_motor_power = left_wheel_velocity
        self.right_motor_power = right_wheel_velocity
        observation, reward, info = self._get_obs(), self._get_reward(), self._get_info()
        self.step_counter += 1
        truncated = True if self.step_counter >= self.truncation_steps else False
        #prev_speed, prev_pitch, prev_yaw, prev_left_speed, prev_right_speed, speed, pitch, yaw, left_speed, right_speed = self._get_obs()
        terminated = True if abs(observation[1]) >= 0.9 or abs(observation[4]) >= 0.25 or abs(observation[5]) > 0.9 else False #or or 
        
        if truncated==True:
            info["is_success"] = True
            reward += 100
        if terminated==True:
            info["is_success"] = False
            reward -= 100
        
        return observation.astype(np.float32), reward, terminated, truncated, info

    def _apply_action(self, action):
        if self._action_type == "discrete":
            if action < len(self.discrete_velocity)/2:
                left_velocity = self.discrete_velocity[action]
                right_velocity = 0
            else:
                left_velocity = 0
                right_velocity = self.discrete_velocity[action]
            return left_velocity, right_velocity

        elif self._action_type == "continuous":
            return (np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity),
                    np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity))
                
    def observation_denormalize(self):
        obs_normalized = self._get_obs()
        obs_denormalized = [
            self._denormalize_1d(value, min_val, max_val)
            for value, min_val, max_val in zip(obs_normalized, self.obs_min, self.obs_max)
        ]
        #print(obs_denormalized)
        return np.array(obs_denormalized)
    
    # Define rotation matrices used to calculate the cameraUpVector according to the movement of the mobile robot
    def _Rx(self,theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    def _Ry(self,theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    def _Rz(self,theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    #sensor readings
    def _get_current_angle(self):
        # Get robot state
        pos, orn = pybullet.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orn)
        linear_velocity, angular_velocity = pybullet.getBaseVelocity(self.robot_id)
        #roll_deg = np.degrees(roll)
        #pitch_deg = np.degrees(pitch)
        #yaw_deg = np.degrees(yaw)
        #angle_degree = pitch_deg
        angle = pitch
        return angle

    def _get_current_line_position(self):
        # Camera processing for line-following
        camera_link_pose = pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[0]
        camera_target_link_pose = pybullet.getLinkState(self.robot_id, self.CAMERA_TARGET_IDX)[0]

        mobile_robot_roll, mobile_robot_pitch, mobile_robot_yaw = pybullet.getEulerFromQuaternion(pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[1])
        R = self._Rz(np.deg2rad(90.0) + mobile_robot_yaw) @ self._Ry(mobile_robot_pitch) @ self._Rx(mobile_robot_roll)
        rotate_camera_up_vector = R @ self.camera_up_vector

        view_matrix = pybullet.computeViewMatrix(cameraEyePosition=camera_link_pose, 
                                             cameraTargetPosition=camera_target_link_pose, 
                                             cameraUpVector=rotate_camera_up_vector)

        width, height, rgb_img, depth_img, seg_img = pybullet.getCameraImage(8, 1, view_matrix, self.projection_matrix)
        img = np.reshape(rgb_img, (height, width, 4))
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        moments = cv2.moments(binary)

        cx = None if moments["m00"] == 0 else moments["m10"] / moments["m00"]
    
        if cx is None:  # Garis tidak terdeteksi
            if self.line_last_position == "left":
                error_steer = -width / 2  # Ujung kiri
            elif self.line_last_position == "right":
                error_steer = width / 2  # Ujung kanan
            else:
                error_steer = self.prev_error_steer  # Tetap posisi terakhir
                
        else:  # Garis terdeteksi
            error_steer = cx - (width / 2)
            # Perbarui posisi terakhir berdasarkan cx
            if cx < width / 2:
                self.line_last_position = "left"
            elif cx > width / 2:
                self.line_last_position = "right"
        self.prev_error_steer = error_steer
        line_position = error_steer
        if cx is None:
            #print(f"\r {cx} ", end="")
            line_position = 4.0
        #print(f"\r {self.prev_error_steer}", end=" ")
        return line_position

    def _get_current_linear_speed(self):
        wheel_radius = 0.045  # 9cm / 2
        left_wheel_velocity = pybullet.getJointState(self.robot_id, self.LEFT_WHEEL_JOINT_IDX)[1]
        right_wheel_velocity = pybullet.getJointState(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX)[1]
        avg_angular_velocity = (left_wheel_velocity + right_wheel_velocity) / 2
        linear_left_wheel = left_wheel_velocity * wheel_radius
        linear_right_wheel = right_wheel_velocity * wheel_radius
        linear_speed = avg_angular_velocity * wheel_radius  # v = ω * r
        return linear_speed, linear_left_wheel, linear_right_wheel

    def _get_first_obs(self):
        speed, left_speed, right_speed = self._get_current_linear_speed()
        pitch = self._get_current_angle()
        line = self._get_current_line_position()
        yaw = line 
        obs = np.array([speed, pitch, yaw, left_speed, right_speed])
        return obs

    def _get_obs(self):
        # pitch, yaw_line, x_dot_motor_linear_speed,  motor_left_speed, motor_right_speed
        speed, left_speed, right_speed = self._get_current_linear_speed()
        pitch = self._get_current_angle()
        line = self._get_current_line_position()
        yaw = line

        obs = np.array([self.target_speed, speed, left_speed, right_speed, pitch, yaw])
        
        #obs_temp = np.array([speed, pitch, yaw, left_speed, right_speed])
        #obs = np.concatenate((self.previous_state, obs_temp))
        
        #self.previous_state = obs_temp
        return self._normalize_obs(obs)

    def _normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _denormalize_1d(self, value, min_value, max_value):
        return value * (max_value - min_value) / 2 + (max_value + min_value) / 2

    def _get_reward(self):

        # +-25 m/s , +- 90 degree(+-pi rad), +-4 units, left wheel, and right wheel ~ in  +-255 rad/s ( +- 40.6 rps or +- 1200 rpm) 
        # 1 = 25               1 = 180          1   =  4
        # 0.5 = 12.5           0.5 = 90         0.5 =  2
        # 0.1 = 2.5            0.1 = 18         0.1 =  1
        # 0.05 = 1.25          0.05 = 9         0.05 = 0.2
        # 0.01 = 0.125 m/s     0.01 = 1.8 deg   0.01 = 0.1 unit
        
        #max error   0.05 (1.25 m/s),  0.05 (9 deg), 0.5 (8 unit) 
        #0.95 = 90.25, 0.25= 6.25 ,0.95 = 0.9025
        #0.05 = 0.25, 0.05= 0.25 , 0.5 = 0.25
        
        #self.state_limit = np.array([25.0, np.pi, 4.0, 25.0, 25.0, 25.0, np.pi, 4.0, 25.0, 25.0])
        #prev_speed, prev_pitch, prev_yaw, prev_left_speed, prev_right_speed, speed, pitch, yaw, left_speed, right_speed = self._get_obs()
        
        target_speed, speed, left_speed, right_speed, pitch, yaw = self._get_obs()
        reward = 0
        
        # Reward  or penalty by pitch and yaw stability
        reward += 1.0 - abs(pitch * 10)**2
        reward += 1.0 - ((abs(yaw * 10)**2)/20)

        reward += 0.25 if left_speed > 0 and right_speed > 0 else 0.0
        reward += 0.25 if left_speed > target_speed else 0.0
        reward += 0.25 if right_speed > target_speed else 0.0
        
            
        #error_speed = self.target_speed - speed
        #error_left_speed = self.target_speed - left_speed
        #error_right_speed = self.target_speed - right_speed

        #reward += 1.0 - (abs(error_left_speed - error_right_speed) * 10)
        #reward += 1.0 - (abs(self.target_speed - speed) * 10)

        
        
        # Penalize large deviations from target values        
        #reward += abs(error_speed)**2 
        #reward += -1 if left_speed < (self.target_speed * 50) else -abs(error_left_speed )**2 
        #reward += -1 if right_speed < (self.target_speed * 50) else -abs(error_right_speed)**2

        #reward -= abs(pitch * 10)**2 
        #reward -= abs(yaw)**2
       
        #reward for near the target
        #reward += (self.target_speed - error_speed) if speed > (self.target_speed * 50) else 0.0
        #reward += (self.target_speed - error_left_speed) if left_speed > (self.target_speed * 50) else 0.0
        #reward += (self.target_speed - error_right_speed) if right_speed > (self.target_speed * 50) else 0.0

        # Small rewards for stable conditions
        #reward += 0.25 - abs(error_speed) if abs(error_speed) < (self.target_speed * 0.75) else abs(error_speed)
        #reward += 0.25 - abs(pitch*10)**2 if abs(pitch) < 0.05 else -abs(pitch*10)**2
        #reward += 0.25 - abs(yaw)**2 if abs(yaw) < 0.5 else -abs(yaw)**2


        # Differential reward based on approaching or moving away from zero
        # Apply differential rewards
        #reward += 0.2 if abs(self.target_speed - speed) < abs(self.target_speed - prev_speed) else -0.1
        #reward += 0.2 if abs(self.target_speed - left_speed) < abs(self.target_speed - prev_left_speed) else -0.1
        #reward += 0.2 if abs(self.target_speed - right_speed) < abs(self.target_speed - prev_right_speed) else -0.1
        #reward += 0.1 if abs(pitch) < abs(prev_pitch) else -0.05
        #reward += 0.1 if abs(yaw) < abs(prev_yaw) else -0.05

        
        # Penalize large deviations from target values
        #reward_1 = -abs(error_speed)**2
        #reward += -(abs(error_speed * 10) **2)/5
        #reward -= min((abs(error_left_speed * 10)**2)/5, 0.5)
        #reward -= min((abs(error_right_speed * 10)**2)/5, 0.5)
        #reward -= min((abs(pitch *10)**2), 0.5)
        #reward -= min((abs(yaw *10)**2), 1.0)

        #reward += 0.125 if abs(error_left_speed) < 0.005 else -0.25
        #reward += 0.125 if abs(error_right_speed) < 0.005 else -0.25

        #reward += 0.125 if left_speed > 0.005 else -0.25
        #reward += 0.125 if right_speed > 0.005 else -0.25

        #reward += 0.25 if abs(pitch) < 0.05 else -0.75
        #reward += 0.25 if abs(yaw) < 0.5 else -0.75

        #reward += 0.2 if abs(pitch) < 0.05 else 0.0

        #reward -= (abs(error_speed * 10) **2)
        #reward -= (abs(error_left_speed * 4) **2)
        #reward -= (abs(error_right_speed * 4) **2)

        
        

        #reward += 0.01 if abs(error_left_speed) < 0.01 else 0
        #reward += 0.01 if abs(error_right_speed) < 0.01 else 0
        #reward += 0.25 if speed > 0.0 and speed < self.target_speed else -0.25
        #reward += 0.25 if left_speed > 0.0 and left_speed < self.target_speed else -0.25
        #reward += 0.25 if right_speed > 0.0 and right_speed < self.target_speed else -0.25

        #reward += 0.25 if abs(pitch) < 0.05 else -0.25
        #reward += 0.25 if abs(yaw) < 0.5 else -0.25

        
        # Differential reward based on approaching or moving away from zero
        # Apply differential rewards
        #reward += 0.1 if abs(self.target_speed - speed) < abs(self.target_speed - prev_speed) else -0.1
        #reward += 0.1 if abs(self.target_speed - left_speed) < abs(self.target_speed - prev_left_speed) else -0.1
        #reward += 0.1 if abs(self.target_speed - right_speed) < abs(self.target_speed - prev_right_speed) else -0.1
        #reward += 0.1 if abs(pitch) < abs(prev_pitch) else -0.1
        #reward += 0.1 if abs(yaw) < abs(prev_yaw) else -0.1

        # Small rewards for stable conditions
        #reward += 0.5 if speed > 0 else -0.5
        #reward += 0.5 - abs(error_left_speed * 10) if abs(error_left_speed) < 0.001 else -(abs(error_left_speed) * 100)**2 / 100
        #reward += 0.5 - abs(error_right_speed * 10) if abs(error_right_speed) < 0.001 else -(abs(error_right_speed) * 100)**2 / 100
        #reward += 0.5 - abs(pitch * 10) if abs(pitch) < 0.1 else -(abs(pitch) * 100)**2 / 100
        #reward += 0.5 - abs(yaw * 10) if abs(yaw) < 0.25 else -(abs(yaw) * 10)**2 / 100
        
        #reward += 0.1 if abs(error_speed) < 0.0001 else -0.05
        #reward += 0.1 if abs(error_left_speed) < 0.0001 else -0.05
        #reward += 0.1 if abs(error_right_speed) < 0.0001 else -0.05
        #reward += 0.1 if abs(pitch) < 0.1 else -0.05
        #reward += 0.1 if abs(yaw) < 0.5 else -0.05

        if self.render_mode == "human":
            print(f"\r State : {speed} | {left_speed} | {right_speed} | {pitch} | {yaw}: >>>>>>>>", end="")

        return reward

    def _get_info(self):
        return {"step_count": self.step_counter}

    def render(self):
        if self.render_mode == "human" and self._physics_client_id >= 0 and not self.rendered_status:
            #self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            #self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
            self.rendered_status = True
            return None

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
