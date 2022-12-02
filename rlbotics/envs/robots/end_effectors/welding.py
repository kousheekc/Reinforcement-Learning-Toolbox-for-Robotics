import numpy as np


class WeldingTorch:
	def __init__(self):
		self.ee_idx = 1
		self.initial_joint_positions = []

		self.jointPivotXYZInParent = [0, 0, 0]
		self.jointPivotRPYInParent = [0, np.pi/2, 0]

		self.jointPivotXYZInChild = [0, 0, 0]
		self.jointPivotRPYInChild = [0, 0, 0]


class SpotWeldingGun:
	def __init__(self):
		self.ee_idx = 1
		self.initial_joint_positions = []

		self.jointPivotXYZInParent = [0, 0, 0]
		self.jointPivotRPYInParent = [0, 0, 0]

		self.jointPivotXYZInChild = [0, 0, 0]
		self.jointPivotRPYInChild = [0, 0, 0]
