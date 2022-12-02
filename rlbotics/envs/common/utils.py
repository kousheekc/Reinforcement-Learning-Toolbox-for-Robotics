import numpy as np
import pybullet as p


def draw_frame(pos, orn, line_length=0.1, replacement_ids=(-1, -1, -1)):
    rot_mat = p.getMatrixFromQuaternion(orn)
    rot_mat = np.reshape(rot_mat, (3, 3))

    end = np.expand_dims(np.array(pos), axis=1) + np.matmul(rot_mat, np.eye(3, 3) * line_length)

    x_id = p.addUserDebugLine(pos, end[:, 0], [1, 0, 0], lineWidth=5, replaceItemUniqueId=replacement_ids[0])
    y_id = p.addUserDebugLine(pos, end[:, 1], [0, 1, 0], lineWidth=5, replaceItemUniqueId=replacement_ids[1])
    z_id = p.addUserDebugLine(pos, end[:, 2], [0, 0, 1], lineWidth=5, replaceItemUniqueId=replacement_ids[2])
    return x_id, y_id, z_id
