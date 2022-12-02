import os
import yaml
import math
import time
import numpy as np
import pybullet as p
import pybullet_data

from rlbotics.envs.common.utils import draw_frame
from rlbotics.envs.robots.utils import combine_urdf
from rlbotics.envs.common.end_effector_class_dict import end_effector_class_dict


class Manipulator:
	"""
	Manipulator base class for all robots. NOTE: robot = arm + gripper
	"""
	def __init__(self, physics_client, robot_name, initial_pose, gripper_name, arm_ee_link, cam_info):
		"""
		:param physics_client: Current physics server
		:param robot_name: Name of arm/robot. {company_name_} + {model}
		:param initial_pose: Initial pose of arm only
		:param gripper_name: Name of gripper
		:param arm_ee_link: Index of arm link for end-effector
		:param cam_info: Information about the placement of the camera
		"""
		p.setRealTimeSimulation(1, physics_client)      # SEE ABOUT THIS LATER. This is needed to complete motion
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.physics_client = physics_client
		self.robot_initial_joint_positions = initial_pose['initial_joint_positions']
		base_pos, base_orn = initial_pose['base_pos'], initial_pose['base_orn']

		# Load YAML info
		yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot_data.yaml')
		with open(yaml_file, 'r') as stream:
			robot_data = yaml.safe_load(stream)

		arm_joint_info = robot_data[robot_name]['joint_info']
		gripper_joint_info = robot_data[gripper_name]['joint_info']

		self.in_built_gripper = robot_data[robot_name]['in_built_gripper']
		self.arm_dof = arm_joint_info['dof']
		self.gripper_dof = gripper_joint_info['dof']

		self.robot_name = robot_name
		self.gripper_name = gripper_name

		# Create gripper object
		self.gripper = end_effector_class_dict[gripper_name]()

		# Get arm and gripper paths
		if self.in_built_gripper:
			if robot_data[robot_name]['location'] == 'local':
				arm_path = os.path.join('rlbotics', 'envs', 'models', robot_data[robot_name]['relative_path'])
			elif robot_data[robot_name]['location'] == 'pybullet_data':
				arm_path = robot_data[robot_name]['relative_path']
			gripper_path = None

		else:
			if robot_data[robot_name]['location'] == 'local':
				arm_path = os.path.join('rlbotics', 'envs', 'models', robot_data[robot_name]['relative_path'])
			elif robot_data[robot_name]['location'] == 'pybullet_data':
				arm_path = robot_data[robot_name]['relative_path']
				
			if robot_data[gripper_name]['location'] == 'local':
				gripper_path = os.path.join('rlbotics', 'envs', 'models', robot_data[gripper_name]['relative_path'])
			elif robot_data[gripper_name]['location'] == 'pybullet_data':
				gripper_path = robot_data[gripper_name]['relative_path']

		# Check if robot data is provided in YAML file. Otherwise get data from urdf
		arm_data_urdf, gripper_data_urdf = self._get_data_from_urdf(arm_path, gripper_path)

		for key, value in arm_joint_info.items():
			if key == 'dof':
				continue
			if value is None:
				arm_joint_info[key] = arm_data_urdf[key]

		for key, value in gripper_joint_info.items():
			if key == 'dof':
				continue
			if value is None:
				gripper_joint_info[key] = gripper_data_urdf[key]

		# Add extra pieces of data from urdf to joint_info dict
		arm_joint_info['joint_friction'] = arm_data_urdf['joint_friction']
		arm_joint_info['joint_damping'] = arm_data_urdf['joint_damping']
		arm_joint_info['joint_max_force'] = arm_data_urdf['joint_max_force']
		gripper_joint_info['joint_friction'] = gripper_data_urdf['joint_friction']
		gripper_joint_info['joint_damping'] = gripper_data_urdf['joint_damping']
		gripper_joint_info['joint_max_force'] = gripper_data_urdf['joint_max_force']

		# Expand robot_info into attributes
		self.arm_joint_indices = arm_data_urdf['joint_indices']
		self.arm_joint_lower_limits = arm_joint_info['joint_lower_limits']
		self.arm_joint_upper_limits = arm_joint_info['joint_upper_limits']
		self.arm_joint_ranges = arm_joint_info['joint_ranges']
		self.arm_joint_velocity_limits = arm_joint_info['joint_velocity_limits']

		self.ee_idx = gripper_data_urdf['ee_idx']
		self.gripper_joint_indices = gripper_data_urdf['joint_indices']
		self.gripper_joint_lower_limits = gripper_joint_info['joint_lower_limits']
		self.gripper_joint_upper_limits = gripper_joint_info['joint_upper_limits']
		self.gripper_joint_ranges = gripper_joint_info['joint_ranges']
		self.gripper_joint_velocity_limits = gripper_joint_info['joint_velocity_limits']
		self.gripper_initial_joint_positions = self.gripper.initial_joint_positions

		self.initial_joint_positions = self.robot_initial_joint_positions + self.gripper_initial_joint_positions

		# Combine urdf if necessary
		if self.in_built_gripper:
			self.robot_path = arm_path
		else:
			arm_info = {
				'name': self.robot_name,
				'path': arm_path,
				'dof': self.arm_dof,
				'ee_link': arm_ee_link,
				'joint_info': arm_joint_info
			}

			gripper_info = {
				'name': self.gripper_name,
				'path': gripper_path,
				'joint_info': gripper_joint_info,
				'jointPivotXYZInParent': self.gripper.jointPivotXYZInParent,
				'jointPivotRPYInParent': self.gripper.jointPivotRPYInParent,
				'jointPivotXYZInChild': self.gripper.jointPivotXYZInChild,
				'jointPivotRPYInChild': self.gripper.jointPivotRPYInChild
			}
			self.robot_path = combine_urdf(arm_info, gripper_info)

		# Load robot
		self.robot_id = p.loadURDF(self.robot_path, base_pos, base_orn, useFixedBase=True, physicsClientId=self.physics_client)

		# Add debugging frame on the end effector
		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn)

		# Set up camera info
		if cam_info is None:
			self.cam_info = {
				'fov': 70, 'aspect': 1.0, 'near_plane': 0.01, 'far_plane': 100,
				'view_dist': 0.5, 'width': 224, 'height': 224,
				'pos': [-0.02, 0.0, 0.0], 'orn': p.getQuaternionFromEuler([0.0, 0.0, -np.pi/2], physicsClientId=self.physics_client),
				'init_view_vec': [0, 0, 1], 'init_up_vec': [0, 1, 0]
			}
		else:
			self.cam_info = cam_info

	def _get_data_from_urdf(self, arm_path, gripper_path=None):
		temp_client = p.connect(p.DIRECT)

		arm_data = {
			'joint_indices': [],
			'joint_lower_limits': [],
			'joint_upper_limits': [],
			'joint_ranges': [],
			'joint_velocity_limits': [],
			'joint_friction': [],
			'joint_damping': [],
			'joint_max_force': []
		}

		gripper_data = {
			'ee_idx': self.gripper.ee_idx,
			'joint_indices': [],
			'joint_lower_limits': [],
			'joint_upper_limits': [],
			'joint_ranges': [],
			'joint_velocity_limits': [],
			'joint_friction': [],
			'joint_damping': [],
			'joint_max_force': []
		}

		if gripper_path is None:
			robot_id = p.loadURDF(arm_path, physicsClientId=temp_client)

			for idx in range(p.getNumJoints(robot_id, temp_client)):
				joint_info = p.getJointInfo(robot_id, idx, temp_client)
				joint_type = joint_info[2]
				joint_lower_limit = joint_info[8]
				joint_upper_limit = joint_info[9]
				joint_velocity_limit = joint_info[11]
				joint_friction = joint_info[7]
				joint_damping = joint_info[6]
				joint_max_force = joint_info[10]
				if joint_type == p.JOINT_FIXED or joint_velocity_limit == 0:
					continue
				if len(arm_data['joint_indices']) < self.arm_dof:
					arm_data['joint_indices'].append(idx)
					arm_data['joint_lower_limits'].append(joint_lower_limit)
					arm_data['joint_upper_limits'].append(joint_upper_limit)
					arm_data['joint_ranges'].append(joint_upper_limit - joint_lower_limit)
					arm_data['joint_velocity_limits'].append(joint_velocity_limit)
					arm_data['joint_friction'].append(joint_friction)
					arm_data['joint_damping'].append(joint_damping)
					arm_data['joint_max_force'].append(joint_max_force)
				else:
					gripper_data['joint_indices'].append(idx)
					gripper_data['joint_lower_limits'].append(joint_lower_limit)
					gripper_data['joint_upper_limits'].append(joint_upper_limit)
					gripper_data['joint_ranges'].append(joint_upper_limit - joint_lower_limit)
					gripper_data['joint_velocity_limits'].append(joint_velocity_limit)
					gripper_data['joint_friction'].append(joint_friction)
					gripper_data['joint_damping'].append(joint_damping)
					gripper_data['joint_max_force'].append(joint_max_force)

			p.removeBody(robot_id, temp_client)

		else:
			arm_id = p.loadURDF(arm_path, physicsClientId=temp_client)
			gripper_id = p.loadURDF(gripper_path, physicsClientId=temp_client)

			# Offset gripper ee idx to account for combined urdf
			gripper_data['ee_idx'] += p.getNumJoints(arm_id, temp_client) - 1

			for idx in range(p.getNumJoints(arm_id, temp_client)):
				joint_info = p.getJointInfo(arm_id, idx, temp_client)
				joint_type = joint_info[2]
				joint_lower_limit = joint_info[8]
				joint_upper_limit = joint_info[9]
				joint_velocity_limit = joint_info[11]
				joint_friction = joint_info[7]
				joint_damping = joint_info[6]
				joint_max_force = joint_info[10]
				if joint_type == p.JOINT_FIXED or joint_velocity_limit == 0:
					continue
				arm_data['joint_indices'].append(idx)
				arm_data['joint_lower_limits'].append(joint_lower_limit)
				arm_data['joint_upper_limits'].append(joint_upper_limit)
				arm_data['joint_ranges'].append(joint_upper_limit - joint_lower_limit)
				arm_data['joint_velocity_limits'].append(joint_velocity_limit)
				arm_data['joint_friction'].append(joint_friction)
				arm_data['joint_damping'].append(joint_damping)
				arm_data['joint_max_force'].append(joint_max_force)

			for idx in range(p.getNumJoints(gripper_id, temp_client)):
				joint_info = p.getJointInfo(gripper_id, idx, temp_client) 
				joint_type = joint_info[2]
				joint_lower_limit = joint_info[8]
				joint_upper_limit = joint_info[9]
				joint_velocity_limit = joint_info[11]
				joint_friction = joint_info[7]
				joint_damping = joint_info[6]
				joint_max_force = joint_info[10]
				if joint_type == p.JOINT_FIXED or joint_velocity_limit == 0:
					continue
				gripper_data['joint_indices'].append(idx + p.getNumJoints(arm_id, temp_client)) # Offset idx to account for combined urdf
				gripper_data['joint_lower_limits'].append(joint_lower_limit)
				gripper_data['joint_upper_limits'].append(joint_upper_limit)
				gripper_data['joint_ranges'].append(joint_upper_limit - joint_lower_limit)
				gripper_data['joint_velocity_limits'].append(joint_velocity_limit)
				gripper_data['joint_friction'].append(joint_friction)
				gripper_data['joint_damping'].append(joint_damping)
				gripper_data['joint_max_force'].append(joint_max_force)

			p.removeBody(arm_id, temp_client)
			p.removeBody(gripper_id, temp_client)

		p.disconnect(temp_client)
		return arm_data, gripper_data

	def reset(self):
		joint_indices = self.arm_joint_indices + self.gripper_joint_indices
		for pos_idx, joint_idx in enumerate(joint_indices):
			p.resetJointState(self.robot_id, joint_idx, self.initial_joint_positions[pos_idx], physicsClientId=self.physics_client)

		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

	def get_joint_limits(self):
		lower_limits = self.arm_joint_lower_limits + self.gripper_joint_lower_limits
		upper_limits = self.arm_joint_upper_limits + self.gripper_joint_upper_limits
		return lower_limits, upper_limits

	def get_joint_ranges(self):
		ranges = self.arm_joint_ranges + self.gripper_joint_ranges
		return ranges

	def get_joint_positions(self):
		joint_indices = self.arm_joint_indices + self.gripper_joint_indices
		return np.array([j[0] for j in p.getJointStates(self.robot_id, joint_indices, physicsClientId=self.physics_client)])

	def get_cartesian_pose(self, orientation_format='euler'):
		state = p.getLinkState(self.robot_id, self.ee_idx, computeForwardKinematics=True, physicsClientId=self.physics_client)
		pos = list(state[0])
		if orientation_format == 'euler':
			orn = list(p.getEulerFromQuaternion(state[1], physicsClientId=self.physics_client))
		else:
			orn = list(state[1])
		return pos, orn

	def get_image(self):
		height, width, view_dist = self.cam_info['height'], self.cam_info['width'], self.cam_info['view_dist']
		fov, aspect, near_plane, far_plane = self.cam_info['fov'], self.cam_info['aspect'], self.cam_info['near_plane'], self.cam_info['far_plane']
		cam_pos, cam_orn = self.cam_info['pos'], self.cam_info['orn']
		init_view_vec, init_up_vec = self.cam_info['init_view_vec'], self.cam_info['init_up_vec']

		# Get end-effector pose
		_, _, _, _, w_pos, w_orn = p.getLinkState(self.robot_id, self.ee_idx,
												  computeForwardKinematics=True,
												  physicsClientId=self.physics_client)

		# Compute camera frame from end effector frame
		pos, orn = p.multiplyTransforms(w_pos, w_orn, cam_pos, cam_orn, physicsClientId=self.physics_client)

		# Get camera frame rotation matrix from quaternion
		rot_mat = p.getMatrixFromQuaternion(orn, physicsClientId=self.physics_client)
		rot_mat = np.array(rot_mat).reshape(3, 3)

		# Transform vectors based on the camera frame
		view_vec = rot_mat.dot(init_view_vec)
		up_vec = rot_mat.dot(init_up_vec)

		view_matrix = p.computeViewMatrix(
			cameraEyePosition=pos,
			cameraTargetPosition=pos + view_dist * view_vec,
			cameraUpVector=up_vec,
			physicsClientId=self.physics_client
		)

		# Camera parameters and projection matrix
		projection_matrix = p.computeProjectionMatrixFOV(
			fov=fov,
			aspect=aspect,
			nearVal=near_plane,
			farVal=far_plane,
			physicsClientId=self.physics_client
		)

		# Extract camera image
		w, h, rgba_img, depth_img, seg_img = p.getCameraImage(
			width=width,
			height=height,
			viewMatrix=view_matrix,
			projectionMatrix=projection_matrix,
			physicsClientId=self.physics_client
		)
		rgb_img = rgba_img[:, :, :3]
		return rgb_img, depth_img, seg_img

	def set_cartesian_pose(self, pose):
		pos = pose[:3]
		roll, pitch, yaw = list(map(lambda x: x % (2*np.pi), pose[3:]))

		# Map RPY : -pi < RPY <= pi
		eul_orn = [-(2*np.pi - roll) if roll > np.pi else roll,
				   -(2*np.pi - pitch) if pitch > np.pi else pitch,
				   -(2*np.pi - yaw) if yaw > np.pi else yaw]

		orn = p.getQuaternionFromEuler(eul_orn, physicsClientId=self.physics_client)

		joint_lower_limits, joint_upper_limits = self.get_joint_limits()
		joint_ranges = self.get_joint_ranges()
		joint_positions = p.calculateInverseKinematics(self.robot_id, self.ee_idx, pos, orn,
													   joint_lower_limits, joint_upper_limits,
													   joint_ranges, self.initial_joint_positions,
													   maxNumIterations=100, physicsClientId=self.physics_client)
		# Remove gripper positions
		joint_positions = joint_positions[:self.arm_dof]
		target_joint_positions = np.array(joint_positions)
		self.set_joint_positions(target_joint_positions)

	def set_joint_positions(self, target_joint_positions, control_freq=1./240.):
		joint_indices = self.arm_joint_indices

		current_joint_positions = self.get_joint_positions()[:self.arm_dof]
		joint_positions_diff = target_joint_positions - current_joint_positions

		# Compute time to complete motion
		max_total_time = np.max(joint_positions_diff / self.arm_joint_velocity_limits)
		num_timesteps = math.ceil(max_total_time / control_freq)
		delta_joint_positions = joint_positions_diff / num_timesteps

		for t in range(1, num_timesteps+1):
			joint_positions = current_joint_positions + delta_joint_positions * t
			p.setJointMotorControlArray(self.robot_id, joint_indices, p.POSITION_CONTROL,
										targetPositions=joint_positions, physicsClientId=self.physics_client)

			p.stepSimulation(self.physics_client)
			time.sleep(control_freq)

			# Update end effector frame display
			pos, orn = self.get_cartesian_pose('quaternion')
			self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

	def open_gripper(self, width=0.08):
		self.gripper.open_gripper(self.robot_id, self.gripper_joint_indices, self.gripper_joint_velocity_limits,
								  self.physics_client, width)
		time.sleep(1)

		# Update end effector frame display
		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

	def close_gripper(self, width=0.0):
		self.gripper.open_gripper(self.robot_id, self.gripper_joint_indices, self.gripper_joint_velocity_limits,
								  self.physics_client, width)
		time.sleep(1)

		# Update end effector frame display
		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)
