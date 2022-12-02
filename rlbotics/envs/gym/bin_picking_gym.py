import gym
import numpy as np
import pybullet as p
from gym import spaces

from rlbotics.envs.common.domain_randomizer import DomainRandomizer
from rlbotics.envs.worlds.bin_picking_world import BinPickingWorld


class BinPickingGym(BinPickingWorld, gym.Env):
    metadata = {'render.modes': ['rgb', 'rgbd', 'rgbds'],
                'video.frames_per_second': 50}    

    def __init__(self, robot, gripper, render=False, obs_mode='rgb', domain_randomization=True, use_ee_cam=False, num_of_parts=5):
        super().__init__(robot, gripper, render, use_ee_cam, num_of_parts)
        self.domain_randomization = domain_randomization
        self.obs_mode = obs_mode
        self.max_timesteps = 1000
        self.timestep = 0
        self.done = False
        self.num_of_parts = num_of_parts
        self.movement_penalty_constant = 10

        # Initialise environment spaces
        if use_ee_cam:
            if self.obs_mode == 'rgb':
                self.observation_space = spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)
            elif self.obs_mode == 'rgbd':
                self.observation_space = spaces.Box(0.01, 1000, shape=(224, 224, 4), dtype=np.uint16)
            elif self.obs_mode == 'rgbds':
                self.observation_space = spaces.Box(0.01, 1000, shape=(224, 224, 5), dtype=np.uint16)
        else:
            if self.obs_mode == 'rgb':
                self.observation_space = spaces.Box(0, 255, shape=(1, 224, 224, 3), dtype=np.uint8)
            elif self.obs_mode == 'rgbd':
                self.observation_space = spaces.Box(0.01, 1000, shape=(1, 224, 224, 4), dtype=np.uint16)
            elif self.obs_mode == 'rgbds':
                self.observation_space = spaces.Box(0.01, 1000, shape=(1, 224, 224, 5), dtype=np.uint16)

        # Initialise env
        self.domain_randomizer = DomainRandomizer(self.np_random)
        self.reset()

    def reset(self):
        self.done = False
        self.timestep = 0
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.reset_world()
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.timestep += 1
        action = np.clip(action, -1, 1).astype(np.float32)

        # Map to appropriate range according to joint joint_limits
        joint_angles = self._map_action_to_joint_space(action)
        self.arm.set_joint_positions(joint_angles)

        delta_joint_angles = joint_angles - self.arm.get_joint_positions(mode='arm')
        rew = self._get_rew(delta_joint_angles)

        new_obs = self._get_obs()
        return new_obs, rew, self.done, {}

    def render(self, mode='rgb'):
        img = self.get_camera_img()

        rgb, dep, seg = img[0], img[1], img[2]

        if mode == 'rgb':
            return rgb

        elif mode == 'rgbd':
            return np.dstack((rgb, dep))

        elif mode == 'rgbds':
            return np.dstack((rgb, dep, seg))

    def close(self):
        p.disconnect()

    def _get_obs(self):
        img = self.render(mode=self.obs_mode)

        if self.use_ee_cam:
            if self.obs_mode == 'rgb':
                rgb = img
                rgb = self.domain_randomizer.randomize_lighting(rgb) if self.domain_randomization else rgb
                img = rgb

            elif self.obs_mode == 'rgbd':
                rgb, dep = img
                rgb = self.domain_randomizer.randomize_lighting(rgb) if self.domain_randomization else rgb
                img = np.dstack((rgb, dep))

            elif self.obs_mode == 'rgbds':
                rgb, dep, seg = img
                rgb = self.domain_randomizer.randomize_lighting(rgb) if self.domain_randomization else rgb
                img = np.dstack((rgb, dep, seg))

        else:
            if self.obs_mode == 'rgb':
                rgb = img
                rgb = self.domain_randomizer.randomize_lighting(rgb) if self.domain_randomization else rgb
                img = rgb

            elif self.obs_mode == 'rgbd':
                rgb, dep = img
                rgb = self.domain_randomizer.randomize_lighting(rgb) if self.domain_randomization else rgb
                img = np.dstack((rgb, dep))

            elif self.obs_mode == 'rgbds':
                rgb, dep, seg = img
                rgb = self.domain_randomizer.randomize_lighting(rgb) if self.domain_randomization else rgb
                img = np.dstack((rgb, dep, seg))

        return img

    def get_camera_img(self):
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0, 2.5],
            cameraTargetPosition=[0.5, 0, 0.94],
            cameraUpVector=[1, 0, 0]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=70,
            aspect=1.0,
            nearVal=0.01,
            farVal=100
        )

        _, _, rgba_img, depth_img, seg_img = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

        rgb_img = rgba_img[:,:,:3]

        if self.domain_randomization:
            rgb_img = self.domain_randomizer.randomize_lighting(rgb_img)

        return rgb_img, depth_img, seg_img

    def _map_action_to_joint_space(self, action):
        joint_positions = np.zeros_like(action)
        for i in range(self.arm.num_joints):
            val = action[i]
            minimum = self.arm.joint_lower_limits[0]
            maximum = self.arm.joint_upper_limits[1]
            joint_positions[i] = (((val - (-1)) * (maximum - minimum)) / (1 - (-1))) + minimum
        return joint_positions

    def _get_rew(self, action):
        # idxs of all overlapping objects with from tray and to tray including arm etc
        all_overlapping_objects_with_tray_1 = np.array(p.getOverlappingObjects(self.tray_1_min_AABB, self.from_tray_max_AABB))[:, 0]
        all_overlapping_objects_with_tray_2 = np.array(p.getOverlappingObjects(self.tray_2_min_AABB, self.to_tray_max_AABB))[:, 0]

        # idxs of overlapping object with from tray and to tray
        overlapping_objects_with_tray_1 = [id for id in all_overlapping_objects_with_tray_1 if id not in self.other_id]
        overlapping_objects_with_tray_2 = [id for id in all_overlapping_objects_with_tray_2 if id not in self.other_id]

        # print("num of cubes in from tray is ", len(overlapping_objects_with_tray_1))
        # print("num of cubes in to tray is ", len(overlapping_objects_with_tray_2))

        # reward = number of cubes outside from tray + number of cubes in to tray
        reward = self.num_of_parts - len(overlapping_objects_with_tray_1) + len(overlapping_objects_with_tray_2)

        if reward == self.num_of_parts * 2 or self.timestep >= self.max_timesteps:
            self.done = True

            # deduct big amount if max_timesteps exceeded
            if self.timestep >= self.max_timesteps:
                reward -= 200

        # movement penalty
        current_joint_pos = np.array(p.getJointStates(self.arm, list(range(self.arm.num_joints))))[:, 0]

        # penalty = sum(abs(current - target)) * some contstant
        movement_penalty = np.absolute(action - current_joint_pos).sum(axis=0) * self.movement_penalty_constant

        reward -= movement_penalty

        return reward

