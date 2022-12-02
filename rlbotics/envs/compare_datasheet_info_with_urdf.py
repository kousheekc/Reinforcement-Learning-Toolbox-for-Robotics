import os
import math
import pybullet as p
import pybullet_data


robot_paths = os.path.abspath(os.path.join('models', 'robots'))
gripper_paths = os.path.abspath(os.path.join('models', 'robots/end_effectors'))

p.connect(p.DIRECT)


def get_info(company, subfolder, name, where='pybullet', angle_format='degrees'):
	if where == 'pybullet':
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
	elif where == 'robots':
		p.setAdditionalSearchPath(robot_paths)
	elif where == 'gripper':
		p.setAdditionalSearchPath(gripper_paths)

	print(os.path.join(company, subfolder, name + '.urdf'))
	ID = p.loadURDF(os.path.join(company, subfolder, name + '.urdf'))
	num_joints = p.getNumJoints(ID)

	joint_upper_limits, joint_lower_limits, joint_ranges, joint_velocity_limits = [], [], [], []
	revolute_joint_indices, prismatic_joint_indices, fixed_joint_indices = [], [], []

	for idx in range(num_joints):
		joint_info = p.getJointInfo(ID, idx)
		joint_type = joint_info[2]
		if joint_type == p.JOINT_REVOLUTE:
			revolute_joint_indices.append(idx)
		elif joint_type == p.JOINT_PRISMATIC:
			prismatic_joint_indices.append(idx)
		elif joint_type == p.JOINT_FIXED:
			fixed_joint_indices.append(idx)

		joint_lower_limits.append(joint_info[8])
		joint_upper_limits.append(joint_info[9])
		joint_ranges.append(joint_info[9] - joint_info[8])
		joint_velocity_limits.append(joint_info[11])

	print('===================================================')
	print('fixed_joints:', fixed_joint_indices)
	print('revolute_joints:', revolute_joint_indices)
	print('prismatic_joints:', prismatic_joint_indices)
	print()

	if angle_format == 'degrees':
		joint_lower_limits = list(map(math.degrees, joint_lower_limits))
		joint_upper_limits = list(map(math.degrees, joint_upper_limits))
		joint_ranges = list(map(math.degrees, joint_ranges))
		joint_velocity_limits = list(map(math.degrees, joint_velocity_limits))

		joint_lower_limits = list(map(round, joint_lower_limits))
		joint_upper_limits = list(map(round, joint_upper_limits))
		joint_ranges = list(map(round, joint_ranges))
		joint_velocity_limits = list(map(round, joint_velocity_limits))

	print('lower_lims:', joint_lower_limits)
	print('upper_lims:', joint_upper_limits)
	print('ranges:', joint_ranges)
	print('velocity_lims:', joint_velocity_limits)
	print('===================================================')


get_info('fanuc', 'r1000ia', 'r1000ia80f', where='robots', angle_format='degrees')


