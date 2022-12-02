from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.abb import Irb5400, Irb6600

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class BodyInWhiteWorld:
    def __init__(self, robot, gripper, render, use_ee_cam=False):
        self.use_ee_cam = use_ee_cam
        self.gripper = gripper
        self.physics_client = p.connect(p.GUI, options='--background_color_red=0.3 --background_color_green=0.3 --background_color_blue=0.3') if render else p.connect(p.DIRECT)
        self.path = os.path.abspath(os.path.dirname(__file__))
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF(os.path.join(os.path.dirname(self.path), 'models', 'misc', 'world_plane', 'plane.urdf'), physicsClientId=self.physics_client)
        p.resetBasePositionAndOrientation(self.plane_id, [0,0,-0.05], p.getQuaternionFromEuler([0,0,0]))

    def reset_world(self):
        rail_orientation = p.getQuaternionFromEuler([np.pi/2,0,0])
        body_in_white_orientation = p.getQuaternionFromEuler([0,0,np.pi/2])

        rail_visual = p.createVisualShape(
            p.GEOM_MESH,
            meshScale=[0.001] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'body_in_white', 'rail.obj'),
            rgbaColor=[0.5, 0.5, 0.5]
        )

        rail_collision = p.createCollisionShape(
            p.GEOM_MESH,
            meshScale=[0.001] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'body_in_white', 'rail.obj'),
        )

        body_in_white_visual = p.createVisualShape(
            p.GEOM_MESH,
            meshScale=[0.001] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'body_in_white', 'car1.obj'),
            rgbaColor=[0.5, 0.5, 0.5]
        )

        body_in_white_collision = p.createCollisionShape(
            p.GEOM_MESH,
            meshScale=[0.001] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'body_in_white', 'car1.obj'),
        )

        self.rail_1 = p.createMultiBody(
            basePosition=[0.5,0,0],
            baseVisualShapeIndex=rail_visual,
            baseCollisionShapeIndex=rail_collision,
            baseOrientation=rail_orientation,
            baseMass=1000,
            physicsClientId=self.physics_client
        )

        self.rail_2 = p.createMultiBody(
            basePosition=[-0.5,0,0],
            baseVisualShapeIndex=rail_visual,
            baseCollisionShapeIndex=rail_collision,
            baseOrientation=rail_orientation,
            baseMass=1000,
            physicsClientId=self.physics_client
        )

        self.body_in_white = p.createMultiBody(
            basePosition=[0,0,0.19],
            baseVisualShapeIndex=body_in_white_visual,
            baseCollisionShapeIndex=body_in_white_collision,
            baseOrientation=body_in_white_orientation,
            baseMass=1000,
            baseInertialFramePosition=[1, 0, 0],
            physicsClientId=self.physics_client
        )

        self.robot1 = Irb6600(self.physics_client, [-2.5,-0,0], [0,0,0,1], gripper_name=self.gripper)

        # self.robot1.reset()

        p.resetBasePositionAndOrientation(self.rail_1, [0.5,0,0], rail_orientation, physicsClientId=self.physics_client)
        p.resetBasePositionAndOrientation(self.rail_2, [-0.5,0,0], rail_orientation, physicsClientId=self.physics_client)


world = BodyInWhiteWorld('Irb6600', 'spot_welding_gun', render=True)

world.reset_world()

while True:
    time.sleep(1)

