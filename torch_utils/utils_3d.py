import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def project_uv_xyz_cam(uv, M, device):
    # adapted from: https://www.cc.gatech.edu/~hays/compvision/proj3/
    
    N = len(uv)
    uv_homog = torch.cat((uv, torch.ones((1, 13), device=device)),dim=0)
    M_inv= torch.linalg.pinv(M)
    xyz = torch.tensordot(M_inv, uv_homog, dims=([1],[0])).T
    return xyz[:,:3]/xyz[:,-1].unsqueeze(dim=1)

def find_intersection(P0,P1,device):
    # from: https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python/52089867
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf."""
    
    # generate all line direction vectors
    dir_vec = (P1-P0) / torch.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized
    # generate the array of all projectors
    projs = torch.eye(dir_vec.shape[1],device=device) - dir_vec[:,:,np.newaxis]*dir_vec[:,np.newaxis]  # I - n*n.T
    # see fig. 1

    # generate R matrix and q vector
    mat_r = projs.sum(axis=0)
    vec_q = (torch.matmul(projs,P0[:,:,np.newaxis])).sum(axis=0)
    
    # solve the least squares problem for the intersection point p: Rp = q
    intersect_p = torch.linalg.lstsq(mat_r,vec_q, rcond=None)[0]
    return intersect_p.T

def skeleton(x,y,z):
    " Definition of skeleton edges "
    
    # rename joints to identify name and axis
    x_head, x_shoulderR, x_shoulderL, x_elbowR, x_elbowL, x_hipR, x_hipL, \
    x_handR, x_handL, x_kneeR, x_kneeL, x_footR, x_footL = x[0], x[1], x[2], x[3], \
    x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]
    y_head, y_shoulderR, y_shoulderL, y_elbowR, y_elbowL, y_hipR, y_hipL, \
    y_handR, y_handL, y_kneeR, y_kneeL, y_footR, y_footL = y[0], y[1], y[2], y[3], \
    y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12]
    z_head, z_shoulderR, z_shoulderL, z_elbowR, z_elbowL, z_hipR, z_hipL, \
    z_handR, z_handL, z_kneeR, z_kneeL, z_footR, z_footL = z[0], z[1], z[2], z[3],\
    z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12]
    
    # definition of the lines of the skeleton graph
    skeleton=np.zeros((14,3,2))
    skeleton[0,:,:]=[[x_head,x_shoulderR],[y_head,y_shoulderR],[z_head,z_shoulderR]]
    skeleton[1,:,:]=[[x_head,x_shoulderL],[y_head,y_shoulderL],[z_head,z_shoulderL]]
    skeleton[2,:,:]=[[x_elbowR,x_shoulderR],[y_elbowR,y_shoulderR],[z_elbowR,z_shoulderR]]
    skeleton[3,:,:]=[[x_elbowL,x_shoulderL],[y_elbowL,y_shoulderL],[z_elbowL,z_shoulderL]]
    skeleton[4,:,:]=[[x_elbowR,x_handR],[y_elbowR,y_handR],[z_elbowR,z_handR]]
    skeleton[5,:,:]=[[x_elbowL,x_handL],[y_elbowL,y_handL],[z_elbowL,z_handL]]
    skeleton[6,:,:]=[[x_hipR,x_shoulderR],[y_hipR,y_shoulderR],[z_hipR,z_shoulderR]]
    skeleton[7,:,:]=[[x_hipL,x_shoulderL],[y_hipL,y_shoulderL],[z_hipL,z_shoulderL]]
    skeleton[8,:,:]=[[x_hipR,x_kneeR],[y_hipR,y_kneeR],[z_hipR,z_kneeR]]
    skeleton[9,:,:]=[[x_hipL,x_kneeL],[y_hipL,y_kneeL],[z_hipL,z_kneeL]]
    skeleton[10,:,:]=[[x_footR,x_kneeR],[y_footR,y_kneeR],[z_footR,z_kneeR]]
    skeleton[11,:,:]=[[x_footL,x_kneeL],[y_footL,y_kneeL],[z_footL,z_kneeL]]
    skeleton[12,:,:]=[[x_shoulderR,x_shoulderL],[y_shoulderR,y_shoulderL],[z_shoulderR,z_shoulderL]]
    skeleton[13,:,:]=[[x_hipR,x_hipL],[y_hipR,y_hipL],[z_hipR,z_hipL]]
    return skeleton

def plotSingle3Dframe(ax, y_true_pred, c='red', limits=[[-500,500],[-500,500],[0,1500]], plot_lines=True):
    " 3D plot of single frame. Can be both label or prediction "
    x = y_true_pred[:, 0]; y = y_true_pred[:, 1]; z = y_true_pred[:, 2]
    ax.scatter(x, y, z, zdir='z', s=20, c=c, marker='o', depthshade=True)
    
    # plot skeleton
    lines_skeleton=skeleton(x,y,z)
    if plot_lines:
        for l in range(len(lines_skeleton)):
            ax.plot(lines_skeleton[l,0,:],lines_skeleton[l,1,:],lines_skeleton[l,2,:], c)

    ax.set_xlabel('X Label'); ax.set_ylabel('Y Label'); ax.set_zlabel('Z Label'); ax.set_aspect('auto')
    
    # set same scale for all the axis
    x_limits=limits[0]; y_limits=limits[1]; z_limits=limits[2]
    x_range = np.abs(x_limits[1] - x_limits[0]); x_middle = np.mean(x_limits)
    y_range = np.abs(y_limits[1] - y_limits[0]); y_middle = np.mean(y_limits)
    z_range = np.abs(z_limits[1] - z_limits[0]); z_middle = np.mean(z_limits)
    
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*np.max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def calc_3D_mpjpe(predicted, target, mask):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    predicted = predicted.squeeze(dim=1)
    assert predicted.shape == target.shape == mask.shape

    if not mask.all(): 
        final_shape = mask.shape
        mpjpe_per_batch = torch.zeros(final_shape[0])
        
        for d in range(final_shape[0]):
            if not mask[d].all():
                num_row_w_false = int((mask.shape[2]*mask.shape[1] - mask[d].sum())/3)
                predicted_temp = predicted[d][mask[d]].view(final_shape[1]-num_row_w_false, final_shape[2]) 
                target_temp = target[d][mask[d]].view(final_shape[1]-num_row_w_false, final_shape[2]) 

                mpjpe_per_batch[d] = torch.nanmean(torch.norm(predicted_temp - target_temp, dim=-1))
            else:         
                mpjpe_per_batch[d] = torch.nanmean(torch.norm(predicted[d] - target[d], dim=-1))

        return torch.nanmean(mpjpe_per_batch)
    
    else:
        return torch.nanmean(torch.norm(predicted - target, dim=len(target.shape)-1))
