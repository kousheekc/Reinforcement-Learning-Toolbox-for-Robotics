from rlbotics.envs.robots.manipulator import Manipulator


# TODO: Add initial joint positions
class Cr7ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_cr7ia'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Cr7ial(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_cr7ial'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Cr35ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_cr35ia'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Lrmate200i(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_lrmate200i'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Lrmate200ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_lrmate200ib'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Lrmate200ib3l(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_lrmate200ib3l'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Lrmate200ic(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_lrmate200ic'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Lrmate200ic5f(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_lrmate200ic5f'
		ee_link = 5

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Lrmate200ic5h(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_lrmate200ic5h'
		ee_link = 5

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Lrmate200ic5hs(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_lrmate200ic5hs'
		ee_link = 5

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class Lrmate200ic5l(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_lrmate200ic5l'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M6ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m6ib'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M6ib6s(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m6ib6s'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M10ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m10ia'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M10ia7l(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m10ia7l'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M16ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m16ib20'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M20ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m20ia'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M20ia10l(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m20ia10l'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M20ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m20ib25'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M430ia2f(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m430ia2f'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.5, 1.4, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M430ia2p(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m430ia2p'
		ee_link = 7

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.5, 1.4, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M710ic45m(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m710ic45m'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M710ic50(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m710ic50'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.6, 0.3, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M900ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m900ia260l'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class M900ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_m900ib700'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)


class R1000ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, gripper_name, initial_joint_positions=None, cam_info=None):
		robot_name = 'fanuc_r1000ia80f'
		ee_link = 6

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = [0.0, 0.4, 0.0, 0.0, 0.0, 0.0]

		initial_pose = {'base_pos': base_pos, 'base_orn': base_orn, 'initial_joint_positions': initial_joint_positions}
		super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
						 arm_ee_link=ee_link, cam_info=cam_info)
