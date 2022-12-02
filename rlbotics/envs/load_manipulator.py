from rlbotics.envs.robots.universal_robots import UR10, UR5, UR3
from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.abb import Irb6600, Irb5400, Irb4400
from rlbotics.envs.robots.kuka import Kr210, Iiwa14, Kr10

import pybullet as p
import pybullet_data
import numpy as np
import time
from rlbotics.envs.common.utils import draw_frame


class PickingEnv:
    def __init__(self, client):
        self.physics_client = physics_client
        self.target_pose = [2, 0.0, 0.03, 0.0, np.pi, -0.0]
        self.plane = p.loadURDF('plane.urdf', [0.0, 0.0, 0.0], useFixedBase=True)
        self.cube_id = p.loadURDF("cube_small.urdf", self.target_pose[:3])
        #time.sleep(10)

        # debug
        pos, orn = p.getBasePositionAndOrientation(self.cube_id)
        draw_frame(pos, orn)

        # Camera setup
        fov, aspect, near_plane, far_plane = 70, 1.0, 0.01, 100
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=[1.0, 0.0, 0.5],
                                               cameraTargetPosition=[0.5, 0, 0.0],
                                               cameraUpVector=[-1, 0, 0])


def test_IK(robot, client):
    env = PickingEnv(client)
    # Target pose
    target_cart_pose = env.target_pose
    target_cart_pose[2] = 1
    time.sleep(2)
    robot.set_cartesian_pose(target_cart_pose)
    time.sleep(2)
    print(robot.get_cartesian_pose())


physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF('plane.urdf')
p.setRealTimeSimulation(1)
#p.setGravity(0, 0, -9.8)

robot1 = Irb4400(physics_client, [0,0,0], [0,0,0,1], gripper_name='spot_welding_gun')
#robot2 = UR10(physics_client, [0,0,0], [0,0,0,1], gripper_name='robotiq_2f_85')
# # robot3 = KukaIiwa(physics_client, [0,0.5,0], [0,0,0,1], gripper_name='robotiq_2f_85')
# robot4 = Panda(physics_client, [0,0,0], [0,0,0,1])

robot1.reset()
# robot2.reset()
# # robot3.reset()
# robot4.reset()
time.sleep(1)

# Test IK
test_IK(robot1, physics_client)


while True:
    time.sleep(1)
    # robot1.set_cartesian_pose([1, 0, 0.2, 0, 0, 0])
    # robot2.set_cartesian_pose([0.5, 1, 0.2, 0, np.pi, 0])
    # robot4.set_cartesian_pose([0.5, 0, 0.2, 0, np.pi, 0])
