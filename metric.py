import numpy as np
from surface_distance import *
import scipy.ndimage

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet

def compute_dice_coefficient(mask_gt, mask_pred):
        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return 0
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2*volume_intersect / volume_sum

def HD95(seg1, seg2, labels=None):
    ret = []
    for i in labels:
        if i==0:
            continue
        if ((seg1==i).sum()==0) or ((seg2==i).sum()==0):
            ret.append(0)
        else:
            hd95 = compute_robust_hausdorff(compute_surface_distances((seg1==i), (seg2==i), np.ones(3)), 95.)
            ret.append(hd95)
    return np.array(ret)

def Dice(seg1, seg2, labels=None):
    ret = []
    for i in labels:
        if i==0:
            continue
        dice = compute_dice_coefficient((seg1==i), (seg2==i))
        ret.append(dice)
    return np.array(ret)

def LogJacDetStd(flow):
    jac_det = (jacobian_determinant(flow[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det).std()
    return log_jac_det

def MAE(img1, img2):
    return np.mean(np.abs(img1-img2))

def NCC(img1, img2):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    cov12 = np.mean((img1-mean1)*(img2-mean2))
    ncc = cov12/(np.std(img1)*np.std(img2))
    return ncc

def get_metrices(img1, img2, seg1, seg2, flow, labels):
    hd95s = HD95(seg1, seg2, labels=labels)
    hd95_score = np.mean(hd95s)
    dices = Dice(seg1, seg2, labels=labels)
    dice_score = np.mean(dices)
    LogJacStd = LogJacDetStd(flow)
    mae = MAE(img1, img2)
    ncc = NCC(img1, img2)
    ret = {
        "hd95_score":hd95_score,
        "dice_score":dice_score,
        "LogJacStd_score":LogJacStd,
        "mae_score":mae,
        "ncc_score":ncc,
        "hd95s":hd95s,
        "dices":dices,
    }
    return ret




