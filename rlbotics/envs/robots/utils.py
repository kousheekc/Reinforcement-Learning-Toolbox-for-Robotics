import pybullet as p
import xml.etree.ElementTree as ET
from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed


def combine_urdf(arm_info, gripper_info):
	arm_name = arm_info['name']
	arm_path = arm_info['path']
	arm_dof = arm_info['dof']
	arm_ee_link = arm_info['ee_link']
	gripper_name = gripper_info['name']
	gripper_path = gripper_info['path']
	jointPivotXYZInParent = gripper_info['jointPivotXYZInParent']
	jointPivotRPYInParent = gripper_info['jointPivotRPYInParent']
	jointPivotXYZInChild = gripper_info['jointPivotXYZInChild']
	jointPivotRPYInChild = gripper_info['jointPivotRPYInChild']

	# Arm Joint info
	arm_joint_lower_limits = arm_info['joint_info']['joint_lower_limits']
	arm_joint_upper_limits = arm_info['joint_info']['joint_upper_limits']
	arm_joint_velocity_limits = arm_info['joint_info']['joint_velocity_limits']
	arm_joint_friction = arm_info['joint_info']['joint_friction']
	arm_joint_damping = arm_info['joint_info']['joint_damping']
	arm_joint_max_force = arm_info['joint_info']['joint_max_force']

	# Gripper Joint info
	gripper_joint_lower_limits = gripper_info['joint_info']['joint_lower_limits']
	gripper_joint_upper_limits = gripper_info['joint_info']['joint_upper_limits']
	gripper_joint_velocity_limits = gripper_info['joint_info']['joint_velocity_limits']
	gripper_joint_friction = gripper_info['joint_info']['joint_friction']
	gripper_joint_damping = gripper_info['joint_info']['joint_damping']
	gripper_joint_max_force = gripper_info['joint_info']['joint_max_force']

	p0 = bc.BulletClient(connection_mode=p.DIRECT)
	p1 = bc.BulletClient(connection_mode=p.DIRECT)

	arm_id = p1.loadURDF(arm_path, flags=p0.URDF_USE_IMPLICIT_CYLINDER)
	gripper_id = p0.loadURDF(gripper_path)

	arm_link_idx = arm_ee_link

	ed0 = ed.UrdfEditor()
	ed0.initializeFromBulletBody(arm_id, p1._client)
	ed1 = ed.UrdfEditor()
	ed1.initializeFromBulletBody(gripper_id, p0._client)

	parentLinkIndex = arm_link_idx

	new_joint = ed0.joinUrdf(ed1, parentLinkIndex, jointPivotXYZInParent, jointPivotRPYInParent,
							 jointPivotXYZInChild, jointPivotRPYInChild, p0._client, p1._client)
	new_joint.joint_type = p0.JOINT_FIXED

	robot_path = f'rlbotics/envs/models/combined/{arm_name}_{gripper_name}.urdf'
	ed0.saveUrdf(robot_path)

	# Fix urdf:
	# -- Add <limit effort="x" lower="y" upper="z" velocity="v"/> for each non-fixed joint. (effort=max_force)
	# -- Correct line: <dynamics damping="1.0" friction="0.0001"/> and insert correct values.
	# -- Fix colors after combination.
	combined_urdf = ET.parse(robot_path)

	root = combined_urdf.getroot()

	cnt = 0
	for joint in root.iter('joint'):
		if joint.attrib['type'] != 'fixed':
			idx = cnt
			if idx < arm_dof:
				max_force = str(arm_joint_max_force[idx])
				lower_limit = str(arm_joint_lower_limits[idx])
				upper_limit = str(arm_joint_upper_limits[idx])
				velocity_limit = str(arm_joint_velocity_limits[idx])
				damping = str(arm_joint_damping[idx])
				friction = str(arm_joint_friction[idx])
			else:
				idx -= arm_dof
				max_force = str(gripper_joint_max_force[idx])
				lower_limit = str(gripper_joint_lower_limits[idx])
				upper_limit = str(gripper_joint_upper_limits[idx])
				velocity_limit = str(gripper_joint_velocity_limits[idx])
				damping = str(gripper_joint_damping[idx])
				friction = str(gripper_joint_friction[idx])

			joint.attrib['type'] = 'revolute'
			limit = ET.Element('limit')
			limit.attrib['effort'] = max_force
			limit.attrib['lower'] = lower_limit
			limit.attrib['upper'] = upper_limit
			limit.attrib['velocity'] = velocity_limit
			joint.append(limit)
			cnt += 1

			for child in joint.iter():
				if child.tag == 'dynamics':
					child.attrib['damping'] = damping
					child.attrib['friction'] = friction
	# Fix colors in urdf
	for i, material in enumerate(root.iter('material')):
		material.attrib['name'] = str(i)

	combined_urdf.write(robot_path)
	return robot_path
