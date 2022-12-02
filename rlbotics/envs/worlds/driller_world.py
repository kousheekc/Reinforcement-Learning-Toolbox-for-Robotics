import os
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
from gym.utils import seeding

from rlbotics.envs.common.robot_class_dict import robot_class_dict


class DrillerWorld:
	def __init__(self, robot, gripper, render, use_ee_cam=False):
		self.use_ee_cam = use_ee_cam
		self.physics_client = p.connect(p.GUI) if render else p.connect(p.DIRECT)
		self.drill_path = os.path.abspath(os.path.join('..', 'models', 'misc', 'drill'))
		self.plane_path = os.path.abspath(os.path.join('..', 'models', 'misc', 'plane'))
		p.setAdditionalSearchPath(pybullet_data.getDataPath())

		# Load Robot and other objects
		self.drill_base_pos = [0.1, 0, 1.2]
		self.drill_base_orn = p.getQuaternionFromEuler([0, 0, np.pi/2], physicsClientId=self.physics_client)
		table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2], physicsClientId=self.physics_client)

		p.loadURDF('plane.urdf', physicsClientId=self.physics_client)

		self.table_id = p.loadURDF('table/table.urdf', [0, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True,
								   physicsClientId=self.physics_client)
		self.drill_id = p.loadURDF(os.path.join(self.drill_path, 'drill.urdf'), self.drill_base_pos, self.drill_base_orn,
								   globalScaling=1.1, physicsClientId=self.physics_client)
		self.load_robot(robot, gripper)

		self.hole = -1
		self.plane = -1
		self.drillbit_link = 0
		self.drill_holster_link = 1

		self.seed()
		self.reset_world()

	def load_robot(self, robot, gripper):
		arm_base_pos = [-0.6, 0, 0.93]
		arm_base_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.physics_client)

		try:
			self.arm = robot_class_dict[robot](self.physics_client, arm_base_pos, arm_base_orn, gripper_name=gripper)
		except KeyError:
			raise FileNotFoundError(f'{robot} or/and {gripper} is an invalid choice')

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset_world(self):
		self.arm.reset()
		p.removeBody(self.hole, physicsClientId=self.physics_client)
		p.removeBody(self.plane, physicsClientId=self.physics_client)
		p.setGravity(0, 0, 0, physicsClientId=self.physics_client)

		p.resetBasePositionAndOrientation(self.drill_id, self.drill_base_pos, self.drill_base_orn,
										  physicsClientId=self.physics_client)
		self.grab_drill()
		self.generate_plane()
		p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)

	def grab_drill(self):
		pos, orn = p.getLinkState(self.drill_id, self.drill_holster_link, physicsClientId=self.physics_client)[4:6]
		self.arm.open_gripper()

		time.sleep(0.2)
		target_pose = [pos[0], pos[1], pos[2]-0.011, 0, np.pi, 0]
		self.arm.set_cartesian_pose(target_pose)
		time.sleep(0.2)

		self.arm.close_gripper()
		target_pose = [pos[0]-0.2, pos[1], pos[2]+0.8, 0, np.pi, 0]
		self.arm.set_cartesian_pose(target_pose)
		time.sleep(0.2)

	def generate_plane(self):
		# Min Max constraints for drilling on plane
		# min = [-0.2, 0.2, 0]
		# max = [0.2, -0.2, 0]

		plane_height = 1.3
		plane_orientation = [0, 0, 0]
		plane_orientation[0] = self.np_random.uniform(0, np.pi/4)
		plane_orientation[1] = self.np_random.uniform(3*np.pi/4, np.pi)
		plane_orientation[2] = self.np_random.uniform(0, np.pi/2)
		plane_orientation = p.getQuaternionFromEuler(plane_orientation, physicsClientId=self.physics_client)
		plane_scale = [self.np_random.uniform(1, 1.4), self.np_random.uniform(1, 1.4), 1]
		hole_relative_pos = [self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2), 0]
		self.hole_world_pos = np.array(p.multiplyTransforms([0, 0, plane_height], plane_orientation, hole_relative_pos,
															plane_orientation, physicsClientId=self.physics_client)[0])
		self.hole_world_orn = plane_orientation

		plane_visual = p.createVisualShape(
			p.GEOM_MESH,
			meshScale=plane_scale,
			fileName=os.path.join(self.plane_path, 'plane.obj'),
			physicsClientId=self.physics_client
		)

		plane_collision = p.createCollisionShape(
			p.GEOM_MESH,
			meshScale=plane_scale,
			fileName=os.path.join(self.plane_path, 'plane.obj'),
			physicsClientId=self.physics_client
		)

		self.plane = p.createMultiBody(
			basePosition=[0, 0, 1.3],
			baseVisualShapeIndex=plane_visual,
			baseCollisionShapeIndex=plane_collision,
			baseOrientation=plane_orientation,
			physicsClientId=self.physics_client
		)

		hole_visual = p.createVisualShape(
			p.GEOM_MESH,
			rgbaColor=[25, 0, 0, 1],
			visualFramePosition=hole_relative_pos,
			fileName=os.path.join(self.plane_path, 'targetHole.obj'),
			physicsClientId=self.physics_client
		)

		self.hole = p.createMultiBody(
			basePosition=[0, 0, 1.3],
			baseVisualShapeIndex=hole_visual,
			baseOrientation=plane_orientation,
			physicsClientId=self.physics_client
		)

	def get_camera_img(self):
		if self.use_ee_cam:
			return self.arm.get_image()

		view_matrix1 = p.computeViewMatrix(
			cameraEyePosition=[0, 0, 2.5],
			cameraTargetPosition=[0, 0, 0],
			cameraUpVector=[1, 0, 0],
			physicsClientId=self.physics_client
		)

		view_matrix2 = p.computeViewMatrix(
			cameraEyePosition=[-0.2, 1.5, 1.3],
			cameraTargetPosition=[-0.2, 0, 1.4],
			cameraUpVector=[0, 1, 0],
			physicsClientId=self.physics_client
		)

		projection_matrix1 = p.computeProjectionMatrixFOV(
			fov=30,
			aspect=1.0,
			nearVal=0.01,
			farVal=2,
			physicsClientId=self.physics_client
		)

		projection_matrix2 = p.computeProjectionMatrixFOV(
			fov=40,
			aspect=1.0,
			nearVal=0.01,
			farVal=2,
			physicsClientId=self.physics_client
		)

		_, _, rgb_img1, depth_img1, seg_img1 = p.getCameraImage(
			width=224,
			height=224,
			viewMatrix=view_matrix1,
			projectionMatrix=projection_matrix1,
			physicsClientId=self.physics_client
		)

		_, _, rgb_img2, depth_img2, seg_img2 = p.getCameraImage(
			width=224,
			height=224,
			viewMatrix=view_matrix2,
			projectionMatrix=projection_matrix2,
			physicsClientId=self.physics_client
		)

		# Remove alpha channel
		rgb_img1, rgb_img2 = rgb_img1[:, :, :3], rgb_img2[:, :, :3]
		return [(rgb_img1, rgb_img2), (depth_img1, depth_img2), (seg_img1, seg_img2)]

	def get_drillbit_pose(self):
		pos, orn = p.getLinkState(self.drill_id, self.drillbit_link, physicsClientId=self.physics_client)[4:6]
		return pos, orn

	def get_hole_pose(self):
		return self.hole_world_pos, self.hole_world_orn

	def get_drillbit_hole_relative_pose(self):
		world_hole_pos, world_hole_orn = self.get_hole_pose()
		world_drillbit_pos, world_drillbit_orn = self.get_drillbit_pose()
		drillbit_world_pos, drillbit_world_orn = p.invertTransform(world_drillbit_pos, world_drillbit_orn,
																   physicsClientId=self.physics_client)

		drill_hole_pos, drill_hole_orn = p.multiplyTransforms(drillbit_world_pos, drillbit_world_orn,
															  world_hole_pos, world_hole_orn,
															  physicsClientId=self.physics_client)
		return drill_hole_pos, drill_hole_orn

	def get_drillbit_hole_distance_and_angle(self):
		drillbit_hole_pos, drillbit_hole_orn = self.get_drillbit_hole_relative_pose()
		drillbit_hole_distance = np.linalg.norm(drillbit_hole_pos)

		drillbit_vector = np.array([0, 0, 1])
		drillbit_hole_orn_matrix = np.reshape(p.getMatrixFromQuaternion(drillbit_hole_orn, physicsClientId=self.physics_client), (3, 3))
		hole_normal = np.dot(drillbit_hole_orn_matrix, drillbit_vector)

		cos_theta = np.dot(drillbit_vector, hole_normal) / (np.linalg.norm(drillbit_vector) * np.linalg.norm(hole_normal))
		theta = math.degrees(math.acos(cos_theta))

		return drillbit_hole_distance, theta



env = DrillerWorld('panda', '', True)
from rlbotics.envs.common.utils import draw_frame
dist, angle = env.get_drillbit_hole_distance_and_angle()
print(dist, angle)
while True:
	time.sleep(1/240)

