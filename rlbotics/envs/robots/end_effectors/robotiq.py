import pybullet as p
import numpy as np


class Robotiq2f85:
	def __init__(self):
		self.ee_idx = 0
		self.initial_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

		self.jointPivotXYZInParent = [0, 0, 0]
		self.jointPivotRPYInParent = [0, np.pi/2, 0]

		self.jointPivotXYZInChild = [0, 0, 0]
		self.jointPivotRPYInChild = [0, 0, 0]

		if self.ee_idx is None:
			self.create_ee_idx()

	def create_ee_idx(self):
		# Create a constraint by adding a fixed joint
		pass

	def open_gripper(self, gripper_id, joint_indices, joint_velocity_limits, physics_client, width=0.08):
		pass

	def close_gripper(self, gripper_id, joint_indices, joint_velocity_limits, physics_client, width=0.0):
		pass
