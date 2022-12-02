from rlbotics.envs.robots.manipulator import Manipulator


class Irb2400(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
		robot_name = 'abb_irb2400'
		ee_link = 5

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.5, 0.1, 0.0, 1.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
						'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Irb4400(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
		robot_name = 'abb_irb4400l_30_243'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.5, 0.1, 0.0, 1.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
						'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Irb5400(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
		robot_name = 'abb_irb5400'
		ee_link = 7

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
						'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Irb6600(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
		robot_name = 'abb_irb6640'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.0, 0, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
						'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)
