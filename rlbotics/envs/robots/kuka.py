from rlbotics.envs.robots.manipulator import Manipulator


class Iiwa7(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_iiwa7'
        ee_link = 8

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, 0.3, 0.0, -1.2, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Iiwa14(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_iiwa14'
        ee_link = 9

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, 0.3, 0.0, -1.2, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr3(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr3r540'
        ee_link = 5

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -0.8, 1.0, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr5(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr5_arc'
        ee_link = 6

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -0.8, 1.0, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr6r700(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr6r700sixx'
        ee_link = 5

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -0.8, 1.0, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr6r900(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr6r900sixx'
        ee_link = 5

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -0.8, 1.0, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr10(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr10r1100sixx'
        ee_link = 5

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -0.8, 1.0, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr16(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr16_2'
        ee_link = 6

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -0.8, 1.0, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr120(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr120r2500pro'
        ee_link = 6

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -0.8, 1.0, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr150(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr150_2'
        ee_link = 6

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -0.8, 1.0, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)


class Kr210(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None, cam_info=None):
        robot_name = 'kuka_kr210l150'
        ee_link = 6

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, 0.1, 0.3, 0.0, 1.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name,
                         arm_ee_link=ee_link, cam_info=cam_info)
