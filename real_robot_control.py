#!/usr/bin/env python3

import time
from xarm.wrapper import XArmAPI

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import time
import math

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import SAC , PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from evalute_trainning import Lite6PickPlaceEnv

def print_info(arm):
    print('position:', arm.position)
    print('tcp_load:', arm.tcp_load)
    print('tcp_offset:', arm.tcp_offset)
    print('tcp_speed_limit:', arm.tcp_speed_limit)
    print('===========JOINT===========')
    print('angles:', arm.angles)
    print('joint_speed_limit:', arm.joint_speed_limit)

def pre_planned_movement(arm):
    #Linear motion, move to pose [300,0,200,180,0,0]
    arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, speed=100, wait=True)

    #Move to pick the object
    arm.open_lite6_gripper()
    time.sleep(0.5)
    arm.set_position(*[300, 0, 300, 180, 0, 0], wait=True)
    arm.set_position(*[300, 0, 185, 180, 0, 0], wait=True)

    print_info(arm)
    time.sleep(0.5)

    #Set tcp payload
    arm.set_tcp_load(0.3, [0, 0, 30])
    arm.set_state(0)

    #Move to place the object
    arm.set_position(*[300, 0, 300, 180, 0, 0], wait=True)
    arm.set_position(*[300, -150, 300, 180, 0, 0], wait=True)
    arm.set_position(*[300, -150, 200, 180, 0, 0], wait=True)
    arm.close_lite6_gripper()
    time.sleep(0.5)

    arm.set_tcp_load(0, [0, 0, 30])
    arm.set_state(0)
    arm.set_position(*[300, -150, 300, 180, 0, 0], wait=True)
    arm.close_lite6_gripper()
    arm.stop_lite6_gripper()

    #Return to home pose [300,0,200,180,0,0]
    arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, speed=100, wait=True)

def set_initial_pos(arm):
    #Linear motion, move to pose [300,0,200,180,0,0]
    arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, speed=100, wait=True)

if __name__ == "__main__":
    model_path = "cluster_ppo_lite6_pick_place_ppo_final"
    vecnorm_path = "cluster_ppo_vec_normalize.pkl"
    #model_path = "ppo_lite6_pick_place_ppo_final"
    #vecnorm_path = "ppo_vec_normalize.pkl"

    #env = DummyVecEnv([lambda: Lite6PickPlaceEnv(render_mode="human")])
    env = DummyVecEnv([lambda: Lite6PickPlaceEnv()])
    #env = DummyVecEnv([lambda:gym.make("CartPole-v1", render_mode="human")])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    #Instantiation API
    arm = XArmAPI('192.168.1.194')
    time.sleep(0.5)

    #Clean error and warn
    if arm.warn_code != 0:
        arm.clean_warn()
    if arm.error_code != 0:
        arm.clean_error()

    #Enable the robot
    arm.motion_enable(enable=True)

    #Set mode and state
    arm.set_mode(0)
    arm.set_state(0)

    arm.set_vacuum_gripper(on=False)
    set_initial_pos(arm)
    time.sleep(1)
    #arm.set_mode(1)
   
    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        if i==0:
            print_info(arm)
            curr_joint_poses = list(info[0]["joint_poses"])
            arm.set_servo_angle( angle = curr_joint_poses , wait=True, speed=1.5, is_radian=True)
            prev_joint_poses=list(info[0]["joint_poses"])
            arm.set_mode(1)
            arm.set_state(0)
            time.sleep(1)

        
        if action[0][3] > 0:
            arm.set_vacuum_gripper(on=True)
        else:
            arm.set_vacuum_gripper(on=False)

        #joint poses in radian
        curr_joint_poses = list(info[0]["joint_poses"])
        joint_speed= np.max(np.absolute( np.array(curr_joint_poses) - np.array(prev_joint_poses)))*24
        print(joint_speed)
        #arm.set_servo_angle( angle = curr_joint_poses , wait=True, speed=joint_speed, is_radian=True)
        arm.set_servo_angle_j( angles = curr_joint_poses , is_radian=True)
        prev_joint_poses=list(info[0]["joint_poses"])
        
        time.sleep(1.0/24.0)

        if done[0]:
            obs = env.reset()
            break

    arm.set_vacuum_gripper(on=False)
    arm.disconnect()
    env.close()

