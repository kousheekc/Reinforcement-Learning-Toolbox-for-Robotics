import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pandas as pd
from gym.utils import seeding

from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.kuka_iiwa import KukaIiwa


class BinPickingWorld:
    def __init__(self, robot, gripper, render, use_ee_cam=False, num_of_parts=5):
        self.use_ee_cam = use_ee_cam
        self.num_of_parts = num_of_parts
        self.physics_client = p.connect(p.GUI) if render else p.connect(p.DIRECT)
        self.path = os.path.abspath(os.path.dirname(__file__))
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load Robot and other objects
        table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

        self.plane_id = p.loadURDF('plane.urdf', physicsClientId=self.physics_client)

        self.table_id = p.loadURDF('table/table.urdf', [0.5, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True,
                                    physicsClientId=self.physics_client)

        self.load_robot(robot, gripper)

        self.tray_1_id = None
        self.tray_2_id = None

        self.parts_id = []
        self.other_id = [self.table_id, self.arm]

        self.parts_data = pd.read_csv(os.path.join(os.path.dirname(self.path), 'models', 'misc', 'random_objects', 'parts_data.csv'))

        self.seed()
        self.reset_world()
    
    def load_robot(self, robot, gripper):
        arm_base_pos = [0, 0, 0.94]
        arm_base_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.physics_client)

        if robot == 'panda':
            self.arm = Panda(self.physics_client, arm_base_pos, arm_base_orn)

        elif robot == 'kuka_iiwa':
            self.arm = KukaIiwa(self.physics_client, arm_base_pos, arm_base_orn)

        elif robot == 'ur10':
            if gripper == 'robotiq_2f_85':
                pass 	# Load UR10 with this gripper. Same for others
            elif gripper == 'robotiq_2f_140':
                pass
            elif gripper == 'robotiq_3f':
                pass
            else:
                raise FileNotFoundError(f'{gripper} gripper does not exist')

        # End of looking through all robots
        else:
            raise FileNotFoundError(f'{robot} robot does not exist')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_world(self):
        self.arm.reset()

        if self.tray_1_id is not None:
            p.removeBody(self.tray_1_id)
        if self.tray_2_id is not None:
            p.removeBody(self.tray_2_id)

        self.tray_1_pos = [self.np_random.uniform(0.4, 0.9), self.np_random.uniform(0.3, 0.7), 0.94]
        self.tray_2_pos = [self.np_random.uniform(0.4, 0.9), self.np_random.uniform(-0.3, -0.7), 0.94]

        self.tray_1_id = p.loadURDF('tray/traybox.urdf', self.tray_1_pos)
        self.tray_2_id = p.loadURDF('tray/traybox.urdf', self.tray_2_pos)

        # bounding box of from tray and to tray
        self.tray_1_min_AABB, self.from_tray_max_AABB = p.getAABB(self.tray_1_id)
        self.tray_2_min_AABB, self.to_tray_max_AABB = p.getAABB(self.tray_2_id)

        for part in self.parts_id:
            p.removeBody(part)

        self.parts_id = []
        self.other_id.append(self.tray_1_id)
        self.other_id.append(self.tray_2_id)

        # add the random objects in tray 1
        for _ in range(self.num_of_parts):
            self.add_random_object()
            time.sleep(0.1)

    def add_random_object(self):
        object_num = self.np_random.randint(7)
        # object_num = 0

        object_scale = self.parts_data.loc[object_num, 'scale']
        object_mass = self.parts_data.loc[object_num, 'mass']
        object_orientation = p.getQuaternionFromEuler([self.np_random.uniform(0, 2*np.pi), self.np_random.uniform(0, 2*np.pi), self.np_random.uniform(0, 2*np.pi)])

        object_visual = p.createVisualShape(
            p.GEOM_MESH,
            meshScale=[object_scale] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'random_objects', str(object_num) + '.obj'),
            rgbaColor=np.hstack((self.np_random.rand(3), 1))
        )

        object_collision = p.createCollisionShape(
            p.GEOM_MESH,
            meshScale=[object_scale] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'random_objects', str(object_num) + '.obj'),
        )

        self.parts_id.append(p.createMultiBody(
            basePosition=[self.np_random.uniform(self.tray_1_pos[0] - 0.1, self.tray_1_pos[0] + 0.1), self.np_random.uniform(self.tray_1_pos[1] - 0.1, self.tray_1_pos[1] + 0.1), 1.5],
            baseVisualShapeIndex=object_visual,
            baseCollisionShapeIndex=object_collision,
            baseOrientation=object_orientation,
            baseMass=object_mass
        ))

    def get_camera_img(self):
        if self.use_ee_cam:
            return self.arm.get_image()

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0, 2.5],
            cameraTargetPosition=[0.5, 0, 0.94],
            cameraUpVector=[1, 0, 0]
            )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=70,
            aspect=1.0,
            nearVal=0.01,
            farVal=100
            )

        _, _, rgba_img, depth_img, seg_img = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
            )

        rgb_img = rgba_img[:,:,:3]

        return rgb_img, depth_img, seg_img
