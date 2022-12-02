import pybullet as p


class PandaGripper:
	"""
	Gripper for Panda robot ONLY.
	"""
	def __init__(self):
		self.ee_idx = 11
		self.initial_joint_positions = [0.0, 0.0]

		if self.ee_idx is None:
			self.create_ee_idx()

	def create_ee_idx(self):
		# Create a constraint by adding a fixed joint
		pass

	def open_gripper(self, gripper_id, joint_indices, joint_velocity_limits, physics_client, width=0.08):
		width = min(width, 0.08)
		target = [width/2, width/2]
		for i in range(len(target)):
			p.setJointMotorControl2(gripper_id, joint_indices[i], p.POSITION_CONTROL, target[i],
									maxVelocity=joint_velocity_limits[i], physicsClientId=physics_client)

	def close_gripper(self, gripper_id, joint_indices, joint_velocity_limits, physics_client, width=0.0):
		width = max(width, 0.0)
		target = [width/2, width/2]
		for i in range(len(target)):
			p.setJointMotorControl2(gripper_id, joint_indices[i], p.POSITION_CONTROL, target[i],
									maxVelocity=joint_velocity_limits[i], physicsClientId=physics_client)
