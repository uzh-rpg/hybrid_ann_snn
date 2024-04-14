import cv2
import numpy as np
import scipy
import torch

__all__ = [
    'get_heatmaps_steps',
    'decompose_projection_matrix',
    'reproject_xyz_onto_world_coord',
    'get_joints_from_heatmap',
    'project_xyz_onto_camera_coord',
    'compose_projection_matrix',
]


def get_heatmaps_steps(xyz, p_mat, width, height):
    """

    Args
        xyz :
            xyz coordinates as 3XNUM_JOINTS wrt world coord system
        p_mat :
            projection matrix from world to image plane
        width :
            width of the resulting frame
        height :
            height of the resulting frame

    Returns
        xyz wrf image coord system, uv image points of skeleton's joints, uv mask
    """
    M, K = decompose_projection_matrix(p_mat)

    u, v, mask = _project_xyz_onto_image(xyz, p_mat, height, width)
    joints = np.stack((v, u), axis=-1)

    xyz_cam = project_xyz_onto_camera_coord(xyz, M)

    return xyz_cam, joints, mask


def get_heatmap(joints, mask, heigth, width, num_joints=13):
    u, v = joints
    # initialize, fill and smooth the heatmaps
    label_heatmaps = np.zeros((heigth, width, num_joints))
    for fmidx, zipd in enumerate(zip(v, u, mask)):
        if (
            zipd[2] == 1
        ):  # write joint position only when projection within frame boundaries
            label_heatmaps[zipd[0], zipd[1], fmidx] = 1
            # label_heatmaps[:, :, fmidx] = decay_heatmap(label_heatmaps[:, :, fmidx])
    return label_heatmaps


def compose_projection_matrix(K, M):
    """
    Compose intrinsics (K) and extrinsics (M) parameters to get a projection
    matrix
    """
    return torch.matmul(K[:, :3], M)


def decompose_projection_matrix(P):
    """
    QR decomposition of world2imageplane projection matrix

    Args
        P :
            Projection matrix word 2 image plane

    Returns
        M matrix, camera matrix
    """

    Q = P[:3, :3]
    q = P[:, 3]
    U, S = scipy.linalg.qr(np.linalg.inv(Q))
    R = np.linalg.inv(U)
    K = np.linalg.inv(S)
    t = S.dot(q)
    K = K / K[2, 2]

    M = np.concatenate([R, np.expand_dims(t, 1)], axis=1)
    camera = np.concatenate([K, np.zeros((3, 1))], axis=1)

    return M, camera


def _project_xyz_onto_image(xyz, p_mat, width, height):
    """

    Args
        xyz :
            xyz in world coordinate system
        p_mat :
            projection matrix word2cam_plane
        width :
            width of resulting frame
        height :
            height of resulting frame

    Returns
        u, v coordinate of skeleton joints as well as joints mask
    """
    num_joints = xyz.shape[-1]
    xyz_homog = np.concatenate([xyz, np.ones([1, num_joints])], axis=0)
    coord_pix_homog = np.matmul(p_mat, xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]

    u = coord_pix_homog_norm[0]
    v = coord_pix_homog_norm[1]
    # flip v coordinate to match the  image direction
    # v = height - v

    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0
    mask[np.isnan(v)] = 0
    mask[u > width] = 0
    mask[u <= 0] = 0
    mask[v > height] = 0
    mask[v <= 0] = 0

    return u, v, mask


def _project_xyz(xyz, M):
    homog = ensure_homogeneous(xyz, d=3).transpose(1, 0)
    result = torch.matmul(M, homog)
    return result.transpose(1, 0)


def project_xyz_onto_camera_coord(
    xyz: torch.Tensor, M: torch.Tensor, invert_z_axis=True
) -> torch.Tensor:
    """

    Args
        xyz :
            xyz coordinates as NUM_JOINTSx3 wrt world coord
        M :
            word2cam EXTRINSIC matrix

    Returns
        xyz coordinates projected onto cam coordinates system
    """
    # Get xyz w.r.t. camera coord system
    xyz_cam = _project_xyz(xyz, M)

    if invert_z_axis:
        # Note: cam coord system is left-handed; Z is along the negative axis
        xyz_cam[:, 2] *= -1

    return xyz_cam


def reproject_xyz_onto_world_coord(xyz, M, invert_z_axis=True):
    """
        Args:
            M : World to camera projection matrix
            xyz : Skeleton joints as NUM_JOINTSx3

        Returns
            Skeleton joints reprojected in world coord system with shape NUM_JOINTSx3
        """

    Tinv = _rotot_inverse(M)
    Tinv = Tinv.type_as(xyz)

    if invert_z_axis:
        xyz[:, 2] *= -1

    reprojected_result = _project_xyz(xyz, Tinv)

    return reprojected_result


def _rotot_inverse(T):
    Rinv = T[:3, :3].transpose(1, 0)
    Tinv = torch.zeros_like(T)
    Tinv[:3, :3] = Rinv
    Tinv[:, 3] = -torch.matmul(Rinv, T[:, 3])
    return Tinv


def get_joints_from_heatmap(y_pr):
    batch_size = y_pr.shape[0]
    n_joints = y_pr.shape[1]
    device = y_pr.device
    confidence = torch.zeros((batch_size, n_joints), device=device)

    p_coords_max = torch.zeros(
        (batch_size, n_joints, 2), dtype=torch.float32, device=device
    )
    for b in range(batch_size):
        for j in range(n_joints):
            pred_joint = y_pr[b, j]
            max_value = torch.max(pred_joint)
            p_coords_max[b, j] = (pred_joint == max_value).nonzero()[0]
            # Confidence of the joint
            confidence[b, j] = max_value

    return p_coords_max, confidence


def ensure_homogeneous(coords, d):
    if isinstance(coords, np.ndarray):
        coords = torch.tensor(coords)

    if coords.size(-1) == d + 1:
        return coords
    assert coords.size(-1) == d
    return cartesian_to_homogeneous(coords)


def cartesian_to_homogeneous(cart):
    hom = torch.cat([cart, torch.ones_like(cart.narrow(-1, 0, 1))], -1)
    return hom
