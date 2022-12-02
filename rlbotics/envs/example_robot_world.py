import pybullet as p
import pybullet_data
import time
import math
import numpy as np


def draw_frame(pos, orn, line_length=0.1, replacement_ids=(-1, -1, -1)):
    rot_mat = p.getMatrixFromQuaternion(orn)
    rot_mat = np.reshape(rot_mat, (3, 3))

    end = np.expand_dims(np.array(pos), axis=1) + np.matmul(rot_mat, np.eye(3, 3) * line_length)

    x_id = p.addUserDebugLine(pos, end[:, 0], [1, 0, 0], lineWidth=5, replaceItemUniqueId=replacement_ids[0])
    y_id = p.addUserDebugLine(pos, end[:, 1], [0, 1, 0], lineWidth=5, replaceItemUniqueId=replacement_ids[1])
    z_id = p.addUserDebugLine(pos, end[:, 2], [0, 0, 1], lineWidth=5, replaceItemUniqueId=replacement_ids[2])
    return x_id, y_id, z_id


class Panda:
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None):
        #p.setRealTimeSimulation(1)      # SEE ABOUT THIS LATER. This is needed to complete motion
        self.physics_client = physics_client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self.robot_id = p.loadURDF('franka_panda/panda.urdf', base_pos, base_orn, useFixedBase=True, flags=flags,
                                   physicsClientId=self.physics_client)

        # Get joint info
        self.gripper_joint_indices = [9, 10]    # Set this manually if gripper is in built
        self.num_joints = p.getNumJoints(self.robot_id)
        self.revolute_joint_indices, self.prismatic_joint_indices, self.fixed_joint_indices = [], [], []
        self.joint_lower_limits, self.joint_upper_limits, self.joint_ranges = [], [], []
        self.gripper_lower_limits, self.gripper_upper_limits, self.gripper_ranges = [], [], []
        self.arm_velocity_limits, self.gripper_velocity_limits = [], []

        for joint_idx in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx, self.physics_client)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.revolute_joint_indices.append(joint_idx)
            elif joint_type == p.JOINT_PRISMATIC:
                self.prismatic_joint_indices.append(joint_idx)
            elif joint_type == p.JOINT_FIXED:
                self.fixed_joint_indices.append(joint_idx)

            if joint_type != p.JOINT_FIXED:
                self.joint_lower_limits.append(joint_info[8])
                self.joint_upper_limits.append(joint_info[9])
                self.joint_ranges.append(joint_info[9] - joint_info[8])

            if joint_type != p.JOINT_FIXED:
                if joint_idx not in self.gripper_joint_indices:
                    self.arm_velocity_limits.append(joint_info[11])
                else:
                    self.gripper_velocity_limits.append(joint_info[11])

        # 7 Revolute, 2 Prismatic, 3 Fixed
        self.arm_num_dof = 7
        self.gripper_num_dof = 2
        self.end_effector_idx = 11

        # Initial pose
        if initial_joint_positions is not None:
            self.initial_joint_positions = initial_joint_positions
        else:
            self.initial_joint_positions = [0.0, 0.3, 0.0, -1.2, 0.0, 2.0, 0.0, 0.0, 0.0]

        # Add debugging frame on the end effector
        pos, orn = self.get_cartesian_pose('quaternion')
        self.ee_ids = draw_frame(pos, orn)

    def reset(self):
        joint_indices = self.revolute_joint_indices + self.prismatic_joint_indices
        joint_indices.sort()
        for pos_idx, joint_idx in enumerate(joint_indices):
            p.resetJointState(self.robot_id, joint_idx, self.initial_joint_positions[pos_idx], physicsClientId=self.physics_client)

        pos, orn = self.get_cartesian_pose('quaternion')
        self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

    def get_joint_limits(self):
        return self.joint_lower_limits, self.joint_upper_limits

    def get_joint_positions(self, mode='arm'):
        if mode == 'arm':
            joint_indices = self.revolute_joint_indices + self.prismatic_joint_indices
            joint_indices.sort()
            for i in self.gripper_joint_indices:
                joint_indices.remove(i)
        elif mode == 'gripper':
            joint_indices = self.gripper_joint_indices

        return np.array([j[0] for j in p.getJointStates(self.robot_id, joint_indices, physicsClientId=self.physics_client)])

    def get_cartesian_pose(self, orientation_format='euler'):
        state = p.getLinkState(self.robot_id, self.end_effector_idx, computeForwardKinematics=True,
                               physicsClientId=self.physics_client)
        pos = list(state[0])
        if orientation_format == 'euler':
            orn = list(p.getEulerFromQuaternion(state[1], physicsClientId=self.physics_client))
        else:
            orn = list(state[1])
        return pos, orn

    def get_image(self, view_dist=0.5, width=224, height=224):
        # Get end-effector pose
        _, _, _, _, w_pos, w_orn = p.getLinkState(self.robot_id, self.end_effector_idx,
                                                  computeForwardKinematics=True,
                                                  physicsClientId=self.physics_client)
        # Camera frame w.r.t end-effector
        cam_orn = p.getQuaternionFromEuler([0.0, 0.0, -np.pi/2], physicsClientId=self.physics_client)
        cam_pos = [-0.02, 0.0, 0.0]

        # Compute camera frame from end effector frame
        pos, orn = p.multiplyTransforms(w_pos, w_orn, cam_pos, cam_orn, physicsClientId=self.physics_client)

        # Get camera frame rotation matrix from quaternion
        rot_mat = p.getMatrixFromQuaternion(orn, physicsClientId=self.physics_client)
        rot_mat = np.array(rot_mat).reshape(3, 3)

        # Initial camera view direction and up direction
        init_view_vec = [0, 0, 1]
        init_up_vec = [0, 1, 0]

        # Transform vectors based on the camera frame
        view_vec = rot_mat.dot(init_view_vec)
        up_vec = rot_mat.dot(init_up_vec)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=pos + view_dist * view_vec,
            cameraUpVector=up_vec,
            physicsClientId=self.physics_client
        )

        # Camera parameters and projection matrix
        fov, aspect, near_plane, far_plane = 70, 1.0, 0.01, 100
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near_plane,
            farVal=far_plane,
            physicsClientId=self.physics_client
        )

        # Extract camera image
        w, h, rgba_img, depth_img, seg_img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            physicsClientId=self.physics_client
        )
        rgb_img = rgba_img[:, :, :3]
        return rgb_img, depth_img, seg_img

    def set_cartesian_pose(self, pose):
        pos = pose[:3]
        roll, pitch, yaw = list(map(lambda x: x % (2*np.pi), pose[3:]))

        # Map RPY : -pi < RPY <= pi
        eul_orn = [-(2*np.pi - roll) if roll > np.pi else roll,
                   -(2*np.pi - pitch) if pitch > np.pi else pitch,
                   -(2*np.pi - yaw) if yaw > np.pi else yaw]

        orn = p.getQuaternionFromEuler(eul_orn, physicsClientId=self.physics_client)

        joint_positions = p.calculateInverseKinematics(self.robot_id, self.end_effector_idx, pos, orn,
                                                       self.joint_lower_limits, self.joint_upper_limits,
                                                       self.joint_ranges, self.initial_joint_positions,
                                                       maxNumIterations=100, physicsClientId=self.physics_client)
        joint_idx = 0
        target_joint_positions = []
        for i in range(self.num_joints):
            if i in self.fixed_joint_indices:
                continue
            if i in self.gripper_joint_indices:
                joint_idx += 1
            else:
                target_joint_positions.append(joint_positions[joint_idx])
                joint_idx += 1
        target_joint_positions = np.array(target_joint_positions)
        self.set_joint_positions(target_joint_positions)

    def set_joint_positions(self, target_joint_positions, control_freq=1./240.):
        joint_indices = self.revolute_joint_indices + self.prismatic_joint_indices
        joint_indices.sort()
        for i in self.gripper_joint_indices:
            joint_indices.remove(i)

        current_joint_positions = self.get_joint_positions(mode='arm')
        joint_positions_diff = target_joint_positions - current_joint_positions

        # Compute time to complete motion
        max_total_time = np.max(joint_positions_diff / self.arm_velocity_limits)
        num_timesteps = math.ceil(max_total_time / control_freq)
        delta_joint_positions = joint_positions_diff / num_timesteps

        # for i in range(self.arm_num_dof):
        #     p.setJointMotorControl2(self.robot_id, joint_indices[i], p.POSITION_CONTROL, maxVelocity=self.arm_velocity_limits[i],
        #                             targetPosition=target_joint_positions[i], physicsClientId=self.physics_client)
        # for i in range(num_timesteps):
        #     p.stepSimulation()
        #     time.sleep(control_freq)

        for t in range(1, num_timesteps+1):
            joint_positions = current_joint_positions + delta_joint_positions * t
            p.setJointMotorControlArray(self.robot_id, joint_indices, p.POSITION_CONTROL,
                                        targetPositions=joint_positions, physicsClientId=self.physics_client)

            # p.stepSimulation(self.physics_client)
            # time.sleep(control_freq)

            # Update end effector frame display
            pos, orn = self.get_cartesian_pose('quaternion')
            self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

        for i in range(num_timesteps):
            p.stepSimulation(self.physics_client)
            time.sleep(control_freq)

        # for i in range(50):
        #     p.stepSimulation()
        #     # Update end effector frame display
        #     pos, orn = self.get_cartesian_pose('quaternion')
        #     self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

    def open_gripper(self, width=0.08):
        width = min(width, 0.08)
        target = [width/2, width/2]
        joints = [9, 10]
        for i in range(len(joints)):
            p.setJointMotorControl2(self.robot_id, joints[i], p.POSITION_CONTROL, target[i],
                                    maxVelocity=self.gripper_velocity_limits[i], physicsClientId=self.physics_client)
        time.sleep(1)

        # Update end effector frame display
        pos, orn = self.get_cartesian_pose('quaternion')
        self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

    def close_gripper(self, width=0.0):
        width = max(width, 0.0)
        target = [width/2, width/2]
        joints = [9, 10]
        for i in range(len(joints)):
            p.setJointMotorControl2(self.robot_id, joints[i], p.POSITION_CONTROL, target[i],
                                    maxVelocity=self.gripper_velocity_limits[i], physicsClientId=self.physics_client)
        time.sleep(1)

        # Update end effector frame display
        pos, orn = self.get_cartesian_pose('quaternion')
        self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)


class PickingEnv:
    def __init__(self, physics_client):
        self.physics_client = physics_client
        self.plane = p.loadURDF('plane.urdf', [0.0, 0.0, 0.0], useFixedBase=True,)
        self.cube_id = p.loadURDF("cube_small.urdf", [0.5, 0.0, 0.03])

        # debug
        pos, orn = p.getBasePositionAndOrientation(self.cube_id)
        draw_frame(pos, orn)

        # Camera setup
        fov, aspect, near_plane, far_plane = 70, 1.0, 0.01, 100
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=[1.0, 0.0, 0.5],
                                               cameraTargetPosition=[0.5, 0, 0.0],
                                               cameraUpVector=[-1, 0, 0])

    def get_image(self):
        # Extract camera image
        w, h, rgba_img, depth_img, seg_img = p.getCameraImage(width=224, height=224, viewMatrix=self.view_matrix,
                                                              projectionMatrix=self.projection_matrix)
        rgb_img = rgba_img[:, :, :3]
        return rgb_img, depth_img, seg_img


def main():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -0.98)

    # Create plane and cube
    env = PickingEnv(physics_client)

    # Create panda
    panda = Panda(physics_client, [0.0,0.0,0.0], [0,0,0,1])
    panda.reset()
    panda.get_image()

    # Target pose
    target_cart_pose = [0.5, 0.0, 0.08, 0.0, np.pi, -0.0]
    time.sleep(2)
    panda.set_cartesian_pose(target_cart_pose)
    time.sleep(2)
    print(panda.get_cartesian_pose())

    # Open gripper
    panda.open_gripper()
    panda.close_gripper()

    # Get final image
    rgb, _, _ = panda.get_image()
    # env.get_image()

    # dummy to keep window open
    while True:
        time.sleep(0.01)


if __name__ == '__main__':
    main()
