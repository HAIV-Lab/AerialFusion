import numpy as np

def compute_reprojection_error(H, src_pts, dst_pts):
    src_homo = np.column_stack([src_pts, np.ones(len(src_pts))])

    dst_pred_homo = H @ src_homo.T
    if H.shape[0] == 2:
        dst_pred = (dst_pred_homo[:1] / dst_pred_homo[1]).T
    if H.shape[0] == 3:
        dst_pred = (dst_pred_homo[:2] / dst_pred_homo[2]).T

    error = np.mean(np.sum((dst_pred - dst_pts) ** 2, axis=1))

    

    return error
