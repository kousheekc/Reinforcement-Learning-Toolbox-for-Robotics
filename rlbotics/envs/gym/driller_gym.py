import os
import gym
import numpy as np
import pybullet as p
from gym import spaces

from rlbotics.envs.common.domain_randomizer import DomainRandomizer
from rlbotics.envs.worlds.driller_world import DrillerWorld


class DrillerGym(DrillerWorld, gym.Env):
	metadata = {'render.modes': ['rgb', 'rgbd', 'rgbds'],
				'video.frames_per_second': 50}

	def __init__(self, robot, gripper, render=False, obs_mode='rgb', domain_randomization=True, use_ee_cam=False, rew_type='dense'):
		super().__init__(robot, gripper, render, use_ee_cam)
		self.domain_randomization = domain_randomization
		self.obs_mode = obs_mode
		self.rew_type = rew_type
		self.max_timesteps = 1000
		self.timestep = 0
		self.done = False

		# Initialise environment spaces
		self.action_space = spaces.Box(-1, 1, (self.arm.arm_num_dof,), dtype=np.float32)
		if use_ee_cam:
			if self.obs_mode == 'rgb':
				self.observation_space = spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)
			elif self.obs_mode == 'rgbd':
				self.observation_space = spaces.Box(0.01, 1000, shape=(224, 224, 4), dtype=np.uint16)
			elif self.obs_mode == 'rgbds':
				self.observation_space = spaces.Box(0.01, 1000, shape=(224, 224, 5), dtype=np.uint16)
		else:
			if self.obs_mode == 'rgb':
				self.observation_space = spaces.Box(0, 255, shape=(2, 224, 224, 3), dtype=np.uint8)
			elif self.obs_mode == 'rgbd':
				self.observation_space = spaces.Box(0.01, 1000, shape=(2, 224, 224, 4), dtype=np.uint16)
			elif self.obs_mode == 'rgbds':
				self.observation_space = spaces.Box(0.01, 1000, shape=(2, 224, 224, 5), dtype=np.uint16)

		# Initialise env
		self.domain_randomizer = DomainRandomizer(self.np_random)
		self.reset()

	def reset(self):
		self.reset_world()
		self.done = False
		self.timestep = 0
		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

		# Randomize physics constraints, plane texture and drill color
		texture_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'textures')
		self.domain_randomizer.randomize_physics_constraints(self.drill_id)
		self.domain_randomizer.randomize_color(self.drill_id)
		self.domain_randomizer.randomize_texture(texture_path, self.plane)

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
		if self.use_ee_cam:
			rgb, dep, seg = img[0], img[1], img[2]

			if mode == 'rgb':
				return rgb

			elif mode == 'rgbd':
				return rgb, dep

			elif mode == 'rgbds':
				return rgb, dep, seg

		else:
			rgb1, rgb2, dep1, dep2, seg1, seg2 = img[0][0], img[0][1], img[1][0], img[1][1], img[2][0], img[2][1]

			if mode == 'rgb':
				return rgb1, rgb2

			elif mode == 'rgbd':
				return rgb1, rgb2, dep1, dep2

			elif mode == 'rgbds':
				return rgb1, rgb2, dep1, dep2, seg1, seg2

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
				rgb1, rgb2 = img
				rgb1, rgb2 = self.domain_randomizer.randomize_lighting(rgb1, rgb2) if self.domain_randomization else rgb1, rgb2
				img = rgb1, rgb2

			elif self.obs_mode == 'rgbd':
				rgb1, rgb2, dep1, dep2 = img
				rgb1, rgb2 = self.domain_randomizer.randomize_lighting(rgb1, rgb2) if self.domain_randomization else rgb1, rgb2
				img = np.dstack((rgb1, dep1)), np.dstack((rgb2, dep2))

			elif self.obs_mode == 'rgbds':
				rgb1, rgb2, dep1, dep2, seg1, seg2 = img
				rgb1, rgb2 = self.domain_randomizer.randomize_lighting(rgb1, rgb2) if self.domain_randomization else rgb1, rgb2
				img = np.dstack((rgb1, dep1, seg1)), np.dstack((rgb2, dep2, seg2))

		return img

	def _map_action_to_joint_space(self, action):
		joint_positions = np.zeros_like(action)
		for i in range(self.arm.num_joints):
			val = action[i]
			minimum = self.arm.joint_lower_limits[0]
			maximum = self.arm.joint_upper_limits[1]
			joint_positions[i] = (((val - (-1)) * (maximum - minimum)) / (1 - (-1))) + minimum
		return joint_positions

	def _get_rew(self, delta_joint_angles):
		rew = 0

		if self.rew_type == 'dense':
			# Check if drill has dropped
			c1 = len(p.getContactPoints(self.arm.robot_id, self.drill_id, linkIndexA=self.arm.gripper_joint_indices[0],
										physicsClientId=self.physics_client))
			c2 = len(p.getContactPoints(self.arm.robot_id, self.drill_id, linkIndexA=self.arm.gripper_joint_indices[1],
										physicsClientId=self.physics_client))
			if c1 + c2 == 0:
				self.done = True
				rew -= 300

			# Check if drill is touching the table
			if len(p.getContactPoints(self.drill_id, self.table_id, physicsClientId=self.physics_client)) != 0 and not self.done:
				rew -= 200

			# Check if robot is touching the drilling plane
			if len(p.getContactPoints(self.arm.robot_id, self.plane, physicsClientId=self.physics_client)) != 0:
				rew -= 200

			# Compute electricity cost
			electricity_scale = 2
			rew -= np.sum(np.abs(delta_joint_angles) * electricity_scale)

			if not self.done:
				# Get distance and angle between target hole and drillbit
				distance, angle = self.get_drillbit_hole_distance_and_angle()

				# Check if task is performed
				if angle <=0.01 and distance <= 0.001:
					self.done = True
					rew += 500
				distance_scale = 100
				angle_scale = 1
				distance *= distance_scale
				angle *= angle_scale

				rew += 100 - distance
				rew += 100 - angle

		else:
			# Sparse rewards
			# Check if drill has dropped
			c1 = len(p.getContactPoints(self.arm.robot_id, self.drill_id, linkIndexA=self.arm.gripper_joint_indices[0],
										physicsClientId=self.physics_client))
			c2 = len(p.getContactPoints(self.arm.robot_id, self.drill_id, linkIndexA=self.arm.gripper_joint_indices[1],
										physicsClientId=self.physics_client))
			if c1 + c2 == 0:
				self.done = True
				rew = -1

			if not self.done:
				# Get distance and angle between target hole and drillbit
				distance, angle = self.get_drillbit_hole_distance_and_angle()

				# Check if task is performed
				if angle <=0.01 and distance <= 0.001:
					self.done = True
					rew = 1

		if self.timestep == self.max_timesteps:
			timestep_penalty = 100
			rew = -1 if self.rew_type == 'sparse' and not self.done else rew - timestep_penalty
			self.timestep = 0
			self.done = True

		return rew
