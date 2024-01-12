import numpy as np

import torch


####new metrics updated with latest papers convention
##########sparse-2-dense paper release####################
def depth_metrics(input_gt_depth_image,pred_depth_image, verbose=True, get_log = True):
    ###STEP 0 #######################################################
    input_gt_depth = input_gt_depth_image.copy()
    pred_depth = pred_depth_image.copy()

    n = np.sum(input_gt_depth > 1e-3) ####valid gt pixels count             
    idxs = ( (input_gt_depth <= 1e-3) ) ####valid gt pixels indices
    
    pred_depth[idxs] = 1 ### mask to 1 invalid pixels
    input_gt_depth[idxs] = 1 ### mask to 1 invalid pixels   

    print('valid samples:',n,'masked samples:', np.sum(idxs))

    ####STEP 1: compute delta######### FCRN standard
    pred_d_gt = pred_depth / input_gt_depth
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth / pred_depth
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n
    ########################################################################################        

    #####STEP 2: compute mean errors################ OmniDepth,HoHoNet, etc. standard
    
    ##normalize to ground truth max

    input_gt_depth_norm = input_gt_depth / np.max(input_gt_depth)
    pred_depth_norm = pred_depth / np.max(input_gt_depth)     
        
        
    ARD = np.sum(np.abs((pred_depth_norm - input_gt_depth_norm)) / (input_gt_depth_norm) / n)
    SRD = np.sum(((pred_depth_norm - input_gt_depth_norm)** 2) / (input_gt_depth_norm) / n)

    ###FIXME - e basis
    log_pred_norm = np.log(pred_depth_norm)
    log_gt_norm = np.log(input_gt_depth_norm)
    ##log_pred = np.log10(pred_depth_norm)
    ##log_gt = np.log10(input_gt_depth_norm)

    ##RMSE_linear = np.sqrt(((pred_depth - input_gt_depth) ** 2).mean())
    RMSE_linear = np.sqrt(np.sum((pred_depth_norm - input_gt_depth_norm) ** 2) / n) ####FIXME - original without norm in any case
    ##RMSE_log = np.sqrt(((log_pred - log_gt) ** 2).mean())
    if(get_log):
        RMSE_log = np.sqrt(np.sum((log_pred_norm - log_gt_norm) ** 2) / n)   ####FIXME - original without norm in any case 
    else:
        RMSE_log = 0.0
      

    if(verbose):
        print('Threshold_1_25: {}'.format(Threshold_1_25))
        print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
        print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
        print('RMSE_linear: {}'.format(RMSE_linear))
        print('RMSE_log: {}'.format(RMSE_log))
        print('SRD (MSE): {}'.format(SRD))
        print('ARD (MAE): {}'.format(ARD))
        
    return Threshold_1_25,Threshold_1_25_2,Threshold_1_25_3, RMSE_linear,RMSE_log,ARD,SRD

######slicenet old metrics
def standard_metrics(input_gt_depth_image,pred_depth_image, verbose=True, normalize_to_max = False):
    ##########################################################
    input_gt_depth = input_gt_depth_image.copy()
    pred_depth = pred_depth_image.copy()

    n = np.sum(input_gt_depth > 1e-3) ####valid pixels
                        
    ###CHECK mask - meters
    idxs = ( (input_gt_depth <= 1e-3) )
    pred_depth[idxs] = 1
    input_gt_depth[idxs] = 1
    ##pred_depth = pred_depth[idxs]
    ##input_gt_depth = input_gt_depth[idxs] 

    print('valid samples:',n,'masked samples:', np.sum(idxs))

    ####STEP 1: compute delta######### FCRN - metrics
    #######prepare mask
    pred_d_gt = pred_depth / input_gt_depth
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth / pred_depth
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n
    ########################################################################################        

    #####STEP 2: compute mean errors################ OmniDepth
    #####CHECK for invalid pixels or max out of range

    print('max pred - gt',  np.max(pred_depth), np.max(input_gt_depth))

    input_gt_depth_norm = input_gt_depth / np.max(input_gt_depth)
    pred_depth_norm = pred_depth / np.max(pred_depth)     
                 
    
    if(normalize_to_max):
        ARD = np.sum(np.abs((pred_depth_norm - input_gt_depth_norm)) / (input_gt_depth_norm) / n)
        SRD = np.sum(((pred_depth_norm - input_gt_depth_norm)** 2) / (input_gt_depth_norm) / n)

        ###FIXME - e basis
        log_pred_norm = np.log(pred_depth_norm)
        log_gt_norm = np.log(input_gt_depth_norm)
        ##log_pred = np.log10(pred_depth_norm)
        ##log_gt = np.log10(input_gt_depth_norm)

        ##RMSE_linear = np.sqrt(((pred_depth - input_gt_depth) ** 2).mean())
        RMSE_linear = np.sqrt(np.sum((pred_depth_norm - input_gt_depth_norm) ** 2) / n) ####FIXME - original without norm in any case
        ##RMSE_log = np.sqrt(((log_pred - log_gt) ** 2).mean())
        RMSE_log = np.sqrt(np.sum((log_pred_norm - log_gt_norm) ** 2) / n)   ####FIXME - original without norm in any case              
    else:
        ###without norm
        ARD = np.sum(np.abs((pred_depth - input_gt_depth)) / (input_gt_depth) / n)
        SRD = np.sum(((pred_depth - input_gt_depth)** 2) / (input_gt_depth) / n)

        ###FIXME - e basis
        log_pred = np.log(pred_depth)
        log_gt = np.log(input_gt_depth)
        ##log_pred = np.log10(pred_depth)
        ##log_gt = np.log10(input_gt_depth)

        ##RMSE_linear = np.sqrt(((pred_depth - input_gt_depth) ** 2).mean())
        RMSE_linear = np.sqrt(np.sum((pred_depth - input_gt_depth) ** 2) / n)
        ##RMSE_log = np.sqrt(((log_pred - log_gt) ** 2).mean())
        RMSE_log = np.sqrt(np.sum((log_pred - log_gt) ** 2) / n)    
   

    if(verbose):
        print('Threshold_1_25: {}'.format(Threshold_1_25))
        print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
        print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
        print('RMSE_linear: {}'.format(RMSE_linear))
        print('RMSE_log: {}'.format(RMSE_log))
        print('SRD (MRE): {}'.format(SRD))
        print('ARD (MAE): {}'.format(ARD))
        
    return Threshold_1_25,Threshold_1_25_2,Threshold_1_25_3, RMSE_linear,RMSE_log,ARD,SRD

####OLD
def compare_depth(input_gt_depth_image,pred_depth_image, verbose=True, use_norm = True):
    input_gt_depth = input_gt_depth_image.copy()
    pred_depth = pred_depth_image.copy()

    if(use_norm):
        input_gt_depth /= np.max(input_gt_depth) ###we dont need scale normalization without masking
        pred_depth /= np.max(pred_depth) ###we dont need scale normalization without masking
    
    n = np.sum(input_gt_depth > 1e-3)
        
    idxs = (input_gt_depth <= 1e-3)
    pred_depth[idxs] = 1
    input_gt_depth[idxs] = 1

    print('gt samples',n,'invalid to 1', np.sum(idxs))

    pred_d_gt = pred_depth / input_gt_depth
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth / pred_depth
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n

    log_pred = np.log(pred_depth)
    log_gt = np.log(input_gt_depth)

    d_i = log_gt - log_pred

    RMSE_linear = np.sqrt(np.sum((pred_depth - input_gt_depth) ** 2) / n)
    RMSE_log = np.sqrt(np.sum((log_pred - log_gt) ** 2) / n)
    RMSE_log_scale_invariant = np.sum(d_i ** 2) / n + (np.sum(d_i) ** 2) / (n ** 2)
    ARD = np.sum(np.abs((pred_depth - input_gt_depth)) / input_gt_depth) / n
    SRD = np.sum(((pred_depth - input_gt_depth) ** 2) / input_gt_depth) / n

    if(verbose):
        print('Threshold_1_25: {}'.format(Threshold_1_25))
        print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
        print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
        print('RMSE_linear: {}'.format(RMSE_linear))
        print('RMSE_log: {}'.format(RMSE_log))
        print('RMSE_log_scale_invariant: {}'.format(RMSE_log_scale_invariant))
        print('SRD (MRE): {}'.format(SRD))
        print('ARD (MAE): {}'.format(ARD))
        
    return Threshold_1_25,Threshold_1_25_2,Threshold_1_25_3, RMSE_linear, RMSE_log,RMSE_log_scale_invariant,ARD,SRD

def eval_metric_hohonet(pred, gt, dmax):
    gt = torch.FloatTensor(gt).clamp(0.01, dmax)
    pred = torch.FloatTensor(pred).clamp(0.01, dmax)
    mre = ((gt - pred).abs() / gt).mean().item()
    mae = (gt - pred).abs().mean().item()
    rmse = ((gt - pred)**2).mean().sqrt().item()
    rmse_log = ((gt.log10() - pred.log10())**2).mean().sqrt().item()
    log10 = (gt.log10() - pred.log10()).abs().mean().item()

    delta = torch.max(pred/gt, gt/pred)
    delta_1 = (delta < 1.25).float().mean().item()
    delta_2 = (delta < 1.25**2).float().mean().item()
    delta_3 = (delta < 1.25**3).float().mean().item()
    return {
        'mre': mre, 'mae': mae, 'rmse': rmse, 'rmse_log': rmse_log, 'log10': log10,
        'delta_1': delta_1, 'delta_2': delta_2, 'delta_3': delta_3,
        }


