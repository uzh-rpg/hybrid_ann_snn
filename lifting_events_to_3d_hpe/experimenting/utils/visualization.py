"""
Visualization toolbox
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb


def _get_3d_ax():
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    ax.view_init(30, 240)
    return ax


def plot_heatmap(img):
    fig, ax = plt.subplots(ncols=img.shape[0], nrows=1, figsize=(20, 20))
    for i in range(img.shape[0]):
        ax[i].imshow(img[i])
        ax[i].axis('off')
    plt.show()


def plot_skeleton_3d(skeleton_gt, skeleton_pred=None):
    """
        Args:
           M: extrinsic matrix as tensor of shape 4x3
           xyz: torch tensor of shape NUM_JOINTSx3
           pred: torch tensor of shape NUM_JOINTSx3
        """

    ax = _get_3d_ax()
    skeleton_gt.plot_3d(ax, c='red')
    if skeleton_pred is not None:
        skeleton_pred.plot_3d(ax, c='blue')


def plot_2d_from_3d(dvs_frame, gt_skeleton, p_mat, idx, pred_skeleton=None):
    """
        To plot image and 2D ground truth and prediction

        Args:
          dvs_frame: frame as vector (1xWxH)
          sample_gt: gt joints as vector (N_jointsx2)

        """
    dvs_frame = dvs_frame[:,:,0]
    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(dvs_frame)
    H, W = dvs_frame.shape

    gt_joints = gt_skeleton.get_2d_points(p_mat, 346, 260)
    ax.plot(gt_joints[:, 0], gt_joints[:, 1], '.', c='red', label='gt')
    if pred_skeleton is not None:
        pred_joints = pred_skeleton.get_2d_points(p_mat, 346, 260)
        ax.plot(pred_joints[:, 0], pred_joints[:, 1], '.', c='blue', label='pred')

    plt.legend()
    # plt.savefig('idx{}.png'.format(idx))


def plot_skeleton_2d(dvs_frame, gt_joints, pred_joints=None):
    """
        To plot image and 2D ground truth and prediction

        Args:
          dvs_frame: frame as vector (1xWxH)
          sample_gt: gt joints as vector (N_jointsx2)

        """

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(dvs_frame)
    ax.axis('off')
    H, W = dvs_frame.shape
    ax.plot(gt_joints[:, 0], gt_joints[:, 1], '.', c='red')
    if pred_joints is not None:
        ax.plot(pred_joints[:, 0], pred_joints[:, 1], '.', c='blue')
    plt.legend()
