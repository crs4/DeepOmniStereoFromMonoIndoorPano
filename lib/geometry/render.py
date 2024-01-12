import torch
import torch.nn.functional as F

import numpy as np

from geometry.panorama import *

import matplotlib.pyplot as plt

import torchvision.transforms.functional as tv

def depth_distance_weights(depth, max_depth=20.0, z_temp = 2.0):
    ##print('depth',depth.shape)##torch.Size([1, 1, 512, 1024])
    weights = 1.0 / torch.exp(z_temp * depth / max_depth)
    return weights


def weighted_average_splat(depth, weights, epsilon=1e-8):
    zero_weights = (weights <= epsilon).detach().type(depth.dtype)
    return depth / (weights + epsilon * zero_weights)


def splat(values, coords, splatted):
    b, c, h, w = splatted.size()
    
    uvs = coords

    u = uvs[:, 0, :, :].unsqueeze(1)
    v = uvs[:, 1, :, :].unsqueeze(1)
    
    u0 = torch.floor(u)
    u1 = u0 + 1
    v0 = torch.floor(v)
    v1 = v0 + 1

    u0_safe = torch.clamp(u0, 0.0, w-1)
    v0_safe = torch.clamp(v0, 0.0, h-1)
    u1_safe = torch.clamp(u1, 0.0, w-1)
    v1_safe = torch.clamp(v1, 0.0, h-1)

    u0_w = (u1 - u) * (u0 == u0_safe).detach().type(values.dtype)
    u1_w = (u - u0) * (u1 == u1_safe).detach().type(values.dtype)
    v0_w = (v1 - v) * (v0 == v0_safe).detach().type(values.dtype)
    v1_w = (v - v0) * (v1 == v1_safe).detach().type(values.dtype)

    top_left_w = u0_w * v0_w
    top_right_w = u1_w * v0_w
    bottom_left_w = u0_w * v1_w
    bottom_right_w = u1_w * v1_w

    weight_threshold = 1e-3
    top_left_w *= (top_left_w >= weight_threshold).detach().type(values.dtype)
    top_right_w *= (top_right_w >= weight_threshold).detach().type(values.dtype)
    bottom_left_w *= (bottom_left_w >= weight_threshold).detach().type(values.dtype)
    bottom_right_w *= (bottom_right_w >= weight_threshold).detach().type(values.dtype)

    ##print(top_left_w)
    ##print(top_right_w)

    for channel in range(c):
        top_left_values = values[:, channel, :, :].unsqueeze(1) * top_left_w
        top_right_values = values[:, channel, :, :].unsqueeze(1) * top_right_w
        bottom_left_values = values[:, channel, :, :].unsqueeze(1) * bottom_left_w
        bottom_right_values = values[:, channel, :, :].unsqueeze(1) * bottom_right_w
                                
        top_left_values = top_left_values.reshape(b, -1)
        top_right_values = top_right_values.reshape(b, -1)
        bottom_left_values = bottom_left_values.reshape(b, -1)
        bottom_right_values = bottom_right_values.reshape(b, -1)

        ##print(top_left_values,top_right_values,bottom_left_values,bottom_right_values)

        top_left_indices = (u0_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        top_right_indices = (u1_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        bottom_left_indices = (u0_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        bottom_right_indices = (u1_safe + v1_safe * w).reshape(b, -1).type(torch.int64)

        ##print(top_left_indices,top_right_indices,bottom_left_indices,bottom_right_indices)
        
        splatted_channel = splatted[:, channel, :, :].unsqueeze(1)
        splatted_channel = splatted_channel.reshape(b, -1)
        splatted_channel.scatter_add_(1, top_left_indices, top_left_values)
        splatted_channel.scatter_add_(1, top_right_indices, top_right_values)
        splatted_channel.scatter_add_(1, bottom_left_indices, bottom_left_values)
        splatted_channel.scatter_add_(1, bottom_right_indices, bottom_right_values)
    splatted = splatted.reshape(b, c, h, w)


def render(input, depth, translation, max_depth, get_mask = False, masked_img = False, filter_iter = 1, masks_th = 0.9, use_tr_depth = True):  
    ###FIXED mask
    b,c,h,w = input.size()

    ###FIXME
    #movement = torch.norm(translation)

    #print('render move:',movement, 'for input',input.shape)

    #if(movement < 0.001):
    #    print('render: no movement case')
    #    mask1 = torch.ones_like(depth)
    #    return input, mask1

    depth = torch.clamp(depth, 0.01, 32.0)###FIXING no-movement case
                
    target_coords = transform_coords(depth, translation)

    print('render: target coords',target_coords.shape)

    if(use_tr_depth):
        depth = transform_depthmap(depth, translation)      
   
    weights = depth_distance_weights(depth, max_depth) ####estimated from src position
                
    splatted_image = torch.zeros_like(input)
    splatted_weights = torch.zeros_like(depth)
            
    splat(input * weights, target_coords, splatted_image) #####CHECK IT
    ##splat(input, target_coords, splatted_image) 
                   
    splat(weights, target_coords, splatted_weights)
            
    #####DEBUG
    ##splatted_weights = torch.clamp(splatted_weights, 0, 1)
    #plt.figure(101)
    #plt.title('splatted_weights')
    #plt.imshow(splatted_weights.squeeze(0).squeeze(0).cpu().numpy())
    ################
                    
    result = weighted_average_splat(splatted_image, splatted_weights)
        
    ##r_sum = result.sum(axis=1)   
    ##mask = torch.where(r_sum > 1e-3, 1.0, 0.).unsqueeze(1)
            
    mask = torch.where( (splatted_weights > 1e-3), 1.0, 0.)
    ##mask = torch.where( (splatted_weights < 1), 1.0, 0.)

    ##print('mask',mask.shape)
        
    for i in range(filter_iter):        
        mask = F.interpolate(mask, size=(h//2, w//2), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
                
    
    mask = torch.where(mask > masks_th, 1.0, 0.)
       
    if(masked_img):
        result =  mask * result
             
    
    if(get_mask):
        #     
        return result, mask
    else:
        return result
        

def render_from_weights(input, depth, translation):       
    ###FIXED mask
    b,c,h,w = input.size()
        
    #target_coords = transform_coords(depth, translation)

    #if(use_tr_depth):
    #    depth = transform_depthmap(depth, translation)      
   
    #weights = depth_distance_weights(depth, max_depth) ####estimated from src position
        
    splatted_image = torch.zeros_like(input)
    splatted_weights = torch.zeros_like(depth)
        
    splat(input * weights, target_coords, splatted_image) 
    ##splat(input, target_coords, splatted_image) 
        
    splat(weights, target_coords, splatted_weights)

    #####DEBUG
    ##splatted_weights = torch.clamp(splatted_weights, 0, 1)
    #plt.figure(101)
    #plt.title('splatted_weights')
    #plt.imshow(splatted_weights.squeeze(0).squeeze(0).cpu().numpy())
    ################
                    
    result = weighted_average_splat(splatted_image, splatted_weights)

    ##r_sum = result.sum(axis=1)   
    ##mask = torch.where(r_sum > 1e-3, 1.0, 0.).unsqueeze(1)
            
    mask = torch.where( (splatted_weights > 1e-3), 1.0, 0.)
    ##mask = torch.where( (splatted_weights < 1), 1.0, 0.)

    
        
    for i in range(filter_iter):        
        mask = F.interpolate(mask, size=(h//2, w//2), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
                
    
    mask = torch.where(mask > masks_th, 1.0, 0.)
       
    if(masked_img):
        result =  mask * result
            
    
    if(get_mask):
        #     
        return result, mask
    else:
        return result

def get_weights(depth, translation, max_depth, use_tr_depth = True):  
    ###FIXED mask
    b,c,h,w = depth.size()
        
    target_coords = transform_coords(depth, translation)

    if(use_tr_depth):
        depth = transform_depthmap(depth, translation)      
   
    weights = depth_distance_weights(depth, max_depth) ####estimated from src position
    
    splatted_weights = torch.zeros_like(depth)
           
    splat(weights, target_coords, splatted_weights)
        
    return weights, splatted_weights

def simple_translate_gpu(crd, rgb, d, cam=[], depth_margin = 4, get_mapping = False):#### crd: H,W,3 - point cloud, rgb: H,W,3, 
    #####
    H, W = rgb.shape[0], rgb.shape[1]

    d_mask = (d<1e-3)
    ##d = torch.where( (d<1e-3), 1.0, d)
    d[d_mask] = -1
            
    # move coordinates and calculate new depth
    tmp_coord = crd - cam ####translated point cloud
    ##new_d = np.sqrt(np.sum(np.square(tmp_coord), axis=2)) 
    new_d = torch.sqrt(torch.sum(torch.square(tmp_coord), dim=2))  ####new depth  
    
    # normalize: /depth
    new_coord = tmp_coord / new_d.reshape(H,W,1)
    ###new_depth = torch.zeros_like(new_d)###????
    img = torch.zeros_like(rgb)
    
    # backward: 3d coordinate to pano image
    [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]#### H,W for each axis
    
    idx = (new_d>0)##np.where(new_d>0)####[H,W] valid indices

       
    # theta: horizontal angle, phi: vertical angle
    theta = torch.zeros_like(x)
    phi = torch.zeros_like(x)
    u1 = torch.zeros_like(x)
    v1 = torch.zeros_like(x)

    ###Y up
    #theta[idx] = np.arctan2(y[idx], np.sqrt(np.square(x[idx]) + np.square(z[idx])))
    #phi[idx] = np.arctan2(-z[idx], x[idx])

    theta[idx] = torch.atan2(z[idx], torch.sqrt(torch.square(x[idx]) + torch.square(y[idx])))###alt - v H,W
    phi[idx] = torch.atan2(x[idx], y[idx])#####azimuth - u H,W
        
    #u = torch.atan2(x, y)
    #v = torch.atan(z / torch.sqrt(x.pow(2) + y.pow(2)))    
    v1[idx] = (0.5 - theta[idx] / np.pi) * H ####- 0.5  # (1 - np.sin(theta[idx]))*H/2 - 0.5
    u1[idx] = (0.5 - phi[idx]/(2*np.pi))*W ####- 0.5

    v, u = torch.floor(v1).to(torch.int64), torch.floor(u1).to(torch.int64)###(512,1024) - (512,1204)
        
    img = torch.zeros_like(rgb)
    # Mask out
    mask = (new_d > 0) & (H > v) & (v > 0) & (W > u) & (u > 0)

    ##print('disocclusion mask shape',mask.shape)

    v = v[mask] ##(512*1024 - masked)
    u = u[mask] ##(512*1024 - masked)
    new_d = new_d[mask]
    rgb = rgb[mask]

    ###manage occlusions
    # Give smaller depth pixel higher priority
    reorder = torch.argsort(-new_d)

    if(get_mapping):
        reorder_inv = torch.argsort(new_d)####to store matches later
        v2 = v[reorder_inv]
        u2 = u[reorder_inv]

       
    v = v[reorder]
    u = u[reorder]
    rgb = rgb[reorder]
           
    img[v, u] = rgb 
     

    if(get_mapping):
        ##inv_depth_order = np.asarray(reorder.cpu().numpy(),dtype=np.int32)
        ##print('DEBUG reorder', reorder.cpu().numpy())

        trg_u = np.asarray(u2.cpu().numpy(),dtype=np.int32)###FIXME
        trg_v = np.asarray(v2.cpu().numpy(),dtype=np.int32)###FIXME

        #####reorder_2d =  np.unravel_index(reorder.cpu().numpy(), mask.cpu().numpy().shape, order = 'C')

        src_v = np.repeat(np.arange(512)[:, None], 1024, axis=1)##reorder_2d[0]
        src_u = np.repeat(np.arange(1024)[None,:], 512, axis=0)##reorder_2d[1]
                
        src_v = src_v[mask.cpu().numpy()]
        src_u = src_u[mask.cpu().numpy()]

        src_v = src_v[reorder_inv.cpu().numpy()]###FIXME
        src_u = src_u[reorder_inv.cpu().numpy()]###FIXME

        trg_src_stack = np.column_stack((trg_u,trg_v,src_u,src_v))


        ####FIXME - choose unique values
        temp_src = np.column_stack((src_u, src_v))
        temp_trg = np.column_stack((trg_u, trg_v))

        ##su = np.unique(temp_src, axis = 0)
        tu, unique_trg_indices = np.unique(temp_trg, axis = 0, return_index = True)

        su = np.zeros_like(tu)

        su = temp_src[unique_trg_indices]###FIXME

        ##print('src occurrency', temp_src.shape, su.shape)
        ##print('trg occurrency', temp_trg.shape, tu.shape)

        trg_src_stack = np.column_stack((tu,su))###FIXME
        ######################################################

        return img, trg_src_stack
    else:
        return img
                
                                       
    return img

def crop(img, slice_col, slice_w):
        B, C, H, W = img.size()
        ##cx = (W//2)-1
        ##cy = (H//2)-1       
        ##pi = slice_c + W/2
        shift_i = int(slice_col - W//2)###rotate from origin 
        
        img1 = torch.roll(img, (0, -shift_i), dims=(2,3))#### neg - clockwise
                      
        return tv.center_crop(img1, (H, slice_w))


def simple_gpu_renderer(input, depth, translation, get_mapping = False, slice_w = 32, slice_col = -1): ####input: (b,3,h,w);(b,1,h,w);(b,1,3)      
    ###FIXED mask
    B,c,H,W = input.size()

    #####build unit sphere
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T

    _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude
    _phi = 2*np.pi*(0.5 - (_y)/W ) # longtitude

    ####Y up
    #axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    #axis1 = np.sin(_theta).reshape(H, W, 1)
    #axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1)

    ###Z up
    axis0 = torch.FloatTensor((np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1)).to(depth.device)
    axis1 = torch.FloatTensor((np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)).to(depth.device)
    axis2 = torch.FloatTensor((np.sin(_theta)).reshape(H, W, 1)).to(input.device).to(depth.device)
      
    #############################################

    tr_img = []

    tr_mask = []

    tr_sr_m = []

    for i in range(B):
        ##x_t = torch.FloatTensor(np.random.random_sample((1,1,3))) 
                        
        rgb = (input[i:i+1].squeeze(0)).permute(1, 2, 0) ###as H,W,3
        d = (depth[i:i+1].squeeze(0)) ###from 1,1,H,W to 1,H,W
         
                
        ##print(translation[i:i+1].squeeze(0).cpu().numpy())

        cam = translation[i:i+1].squeeze(0) ###as 1,3##np.array([0.2, 0.42, -0.18]).reshape(1,3)#

        cam[:,0] = -cam[:,0]#####to s3d reference

        cam /= torch.max(d)

        d = d.reshape(H, W, 1) / torch.max(d)#####MOD no norm
        d_mask = (d<1e-3)
        ##d = torch.where( (d<1e-3), 1.0, d)
        d[d_mask] = 1
        
        coord = torch.cat((axis0, axis1, axis2), dim=2) * d##### H,W,3 - point cloud

        if(get_mapping):                        
            img1, trg_src_mapping = simple_translate_gpu(coord, rgb, d, cam, get_mapping = get_mapping) #####H,W,3
        else:
            img1 = simple_translate_gpu(coord, rgb, d, cam, get_mapping = get_mapping) #####H,W,3
                
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
                
        mask1 = torch.where(img1>0, 1, 0)

        mask1 = mask1[:,2:]
        
        ##img1 = F.interpolate(img1, size=(H//2, W//2), mode='bilinear', align_corners=False)
        ##img1 = F.interpolate(img1, size=(H, W), mode='bilinear', align_corners=False)
                
        tr_img.append(img1)
        tr_mask.append(mask1)

        if(get_mapping):
            tr_sr_m.append(torch.FloatTensor(trg_src_mapping).unsqueeze(0))


    tr_img = torch.cat(tr_img, dim=0)
    tr_mask = torch.cat(tr_mask, dim=0)

    if(slice_col != -1):
        tr_img = crop(tr_img,slice_col,slice_w)
        tr_mask = crop(tr_mask,slice_col,slice_w)    

    if(get_mapping):
        tr_sr_m = torch.cat(tr_sr_m, dim=0)
        return tr_img, tr_mask, tr_sr_m
    else:
        return tr_img, tr_mask

#####DEBUG - for comparison
def omninerf_translate(crd, rgb, d, cam=[], depth_margin = 4):
    #####
    H, W = rgb.shape[0], rgb.shape[1]


    d = np.where(d==0, -1, d)
    
    # move coordinates and calculate new depth
    tmp_coord = crd - cam
    new_d = np.sqrt(np.sum(np.square(tmp_coord), axis=2))    
    
    # normalize: /depth
    new_coord = tmp_coord / new_d.reshape(H,W,1)
    new_depth = np.zeros(new_d.shape)
    img = np.zeros(rgb.shape)
    
    # backward: 3d coordinate to pano image
    [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]
    

    idx = np.where(new_d>0)
    
    # theta: horizontal angle, phi: vertical angle
    theta = np.zeros(y.shape)
    phi = np.zeros(y.shape)
    x1 = np.zeros(z.shape)
    y1 = np.zeros(z.shape)

    ###Y up
    #theta[idx] = np.arctan2(y[idx], np.sqrt(np.square(x[idx]) + np.square(z[idx])))
    #phi[idx] = np.arctan2(-z[idx], x[idx])

    theta[idx] = np.arctan2(z[idx], np.sqrt(np.square(x[idx]) + np.square(y[idx])))
    phi[idx] = np.arctan2(x[idx], y[idx])

    #u = torch.atan2(x, y)
    #v = torch.atan(z / torch.sqrt(x.pow(2) + y.pow(2)))
    
    x1[idx] = (0.5 - theta[idx] / np.pi) * H #- 0.5  # (1 - np.sin(theta[idx]))*H/2 - 0.5
    y1[idx] = (0.5 - phi[idx]/(2*np.pi))*W #- 0.5
    x, y = np.floor(x1).astype('int'), np.floor(y1).astype('int')
    
    img = np.zeros(rgb.shape)
    # Mask out
    mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)
    x = x[mask]
    y = y[mask]
    new_d = new_d[mask]
    rgb = rgb[mask]
    # Give smaller depth pixel higher priority
    reorder = np.argsort(-new_d)
    x = x[reorder]
    y = y[reorder]
    new_d = new_d[reorder]
    rgb = rgb[reorder]
    # Assign
    new_depth[x, y] = new_d
    img[x, y] = rgb                

    
    for i in range(depth_margin, H, 2):
        for j in range(depth_margin, W, 2):

            x_l = max(0, i-depth_margin)
            x_r = min(H, i+depth_margin)
            y_l, y_r = max(0, j-depth_margin), min(W, j+depth_margin)
            
            index = np.where(new_depth[x_l:x_r, y_l:y_r]>0)
            if len(index[0]) == 0: continue 
                        
            mean = np.median(new_depth[x_l:x_r, y_l:y_r][index]) # median
            target_index = np.where(new_depth[x_l:x_r, y_l:y_r] > mean*1.3)
            
            if len(target_index[0]) > depth_margin ** 2 // 2:
                # reduce block size
                img[x_l:x_r, y_l:y_r][target_index] = 0#np.array([255.0, 0.0, 0.0])
                new_depth[x_l:x_r, y_l:y_r][target_index] = 0
    
    
    mask = (new_depth != 0).astype(int)

    return img, new_depth.reshape(H,W,1), tmp_coord, cam.reshape(1, 1, 3), mask

def omninerf_translate_gpu(crd, rgb, d, cam=[], depth_margin = 4):
    #####
    H, W = rgb.shape[0], rgb.shape[1]

    d_mask = (d<1e-3)
    ##d = torch.where( (d<1e-3), 1.0, d)
    d[d_mask] = -1
            
    # move coordinates and calculate new depth
    tmp_coord = crd - cam
    ##new_d = np.sqrt(np.sum(np.square(tmp_coord), axis=2)) 
    new_d = torch.sqrt(torch.sum(torch.square(tmp_coord), dim=2))    
    
    # normalize: /depth
    new_coord = tmp_coord / new_d.reshape(H,W,1)
    new_depth = torch.zeros_like(new_d)
    img = torch.zeros_like(rgb)
    
    # backward: 3d coordinate to pano image
    [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]
    

    idx = (new_d>0)##np.where(new_d>0)
    
    # theta: horizontal angle, phi: vertical angle
    theta = torch.zeros_like(y)
    phi = torch.zeros_like(y)
    x1 = torch.zeros_like(z)
    y1 = torch.zeros_like(z)

    ###Y up
    #theta[idx] = np.arctan2(y[idx], np.sqrt(np.square(x[idx]) + np.square(z[idx])))
    #phi[idx] = np.arctan2(-z[idx], x[idx])

    theta[idx] = torch.atan2(z[idx], torch.sqrt(torch.square(x[idx]) + torch.square(y[idx])))
    phi[idx] = torch.atan2(x[idx], y[idx])

    #u = torch.atan2(x, y)
    #v = torch.atan(z / torch.sqrt(x.pow(2) + y.pow(2)))
    
    x1[idx] = (0.5 - theta[idx] / np.pi) * H #- 0.5  # (1 - np.sin(theta[idx]))*H/2 - 0.5
    y1[idx] = (0.5 - phi[idx]/(2*np.pi))*W #- 0.5
    x, y = torch.floor(x1).to(torch.int64), torch.floor(y1).to(torch.int64)
    
    img = torch.zeros_like(rgb)
    # Mask out
    mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)
    x = x[mask]
    y = y[mask]
    new_d = new_d[mask]
    rgb = rgb[mask]
    # Give smaller depth pixel higher priority
    reorder = torch.argsort(-new_d)
    x = x[reorder]
    y = y[reorder]
    new_d = new_d[reorder]
    rgb = rgb[reorder]
    # Assign
    new_depth[x, y] = new_d
    img[x, y] = rgb                

    
    for i in range(depth_margin, H, 2):
        for j in range(depth_margin, W, 2):

            x_l = max(0, i-depth_margin)
            x_r = min(H, i+depth_margin)
            y_l, y_r = max(0, j-depth_margin), min(W, j+depth_margin)
            
            index = (new_depth[x_l:x_r, y_l:y_r]>0)

            if len(index[0]) == 0: continue 

            ##print(new_depth[x_l:x_r, y_l:y_r][index])

            ##mean = np.median(new_depth[x_l:x_r, y_l:y_r][index]) # median

            mean = (new_depth[x_l:x_r, y_l:y_r]).quantile(q=0.5)####FIXMEEEEEEEEEEEEEEEEE
            
            target_index = (new_depth[x_l:x_r, y_l:y_r] > mean*1.3)
            
            if len(target_index[0]) > depth_margin ** 2 // 2:
                # reduce block size
                img[x_l:x_r, y_l:y_r][target_index] = 0#np.array([255.0, 0.0, 0.0])
                new_depth[x_l:x_r, y_l:y_r][target_index] = 0
    
    
    mask = (new_depth != 0).to(torch.int64)##.astype(int)

    return img,new_depth.reshape(H,W,1), tmp_coord, cam.reshape(1, 1, 3), mask

def omninerf_renderer(input, depth, translation): ####input: (b,3,h,w);(b,h,w);(b,1,3)      
    ###FIXED mask
    B,c,H,W = input.size()

    #####build unit sphere
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T

    _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude
    _phi = 2*np.pi*(0.5 - (_y)/W ) # longtitude

    ####Y up
    #axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    #axis1 = np.sin(_theta).reshape(H, W, 1)
    #axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1)

    ###Z up
    axis0 = (np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1)
    axis1 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    axis2 = (np.sin(_theta)).reshape(H, W, 1)
      
    #############################################

    tr_img = []

    tr_mask = []

    for i in range(B):
        ##x_t = torch.FloatTensor(np.random.random_sample((1,1,3))) 
                        
        rgb = (input[i:i+1].squeeze(0)).permute(1, 2, 0).cpu().numpy() ###as H,W,3
        d = (depth[i:i+1].squeeze(0)).cpu().numpy() ###as H,W

        ###FIXME translation scale

        ##print(translation[i:i+1].squeeze(0).cpu().numpy())

        cam = translation[i:i+1].squeeze(0).cpu().numpy() ###as 1,3##np.array([0.2, 0.42, -0.18]).reshape(1,3)#

        cam[:,0] = -cam[:,0]#####to s3d reference

        cam /= np.max(d)

        d = d.reshape(rgb.shape[0], rgb.shape[1], 1) / np.max(d)#####MOD no norm
        d = np.where(d==0, 1, d)

        coord = np.concatenate((axis0, axis1, axis2), axis=2)*d

        img1, d1, _, _, mask1 = omninerf_translate(coord, rgb, d, cam)
                
        img1 = torch.FloatTensor(img1).permute(2, 0, 1).unsqueeze(0)
        mask1 = torch.FloatTensor(mask1).unsqueeze(0).unsqueeze(0)
                
        tr_img.append(img1)
        tr_mask.append(mask1)

    tr_img = torch.cat(tr_img, dim=0).to(input.device)
    tr_mask = torch.cat(tr_mask, dim=0).to(input.device)

    return tr_img, tr_mask

def omninerf_renderer_gpu(input, depth, translation): ####input: (b,3,h,w);(b,h,w);(b,1,3)      
    ###FIXED mask
    B,c,H,W = input.size()

    #####build unit sphere
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T

    _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude
    _phi = 2*np.pi*(0.5 - (_y)/W ) # longtitude

    ####Y up
    #axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    #axis1 = np.sin(_theta).reshape(H, W, 1)
    #axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1)

    ###Z up
    axis0 = torch.FloatTensor((np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1)).to(depth.device)
    axis1 = torch.FloatTensor((np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)).to(depth.device)
    axis2 = torch.FloatTensor((np.sin(_theta)).reshape(H, W, 1)).to(input.device).to(depth.device)
      
    #############################################

    tr_img = []

    tr_mask = []

    for i in range(B):
        ##x_t = torch.FloatTensor(np.random.random_sample((1,1,3))) 
                        
        rgb = (input[i:i+1].squeeze(0)).permute(1, 2, 0) ###as H,W,3
        d = (depth[i:i+1].squeeze(0)) ###as H,W

        ###FIXME translation scale

        ##print(translation[i:i+1].squeeze(0).cpu().numpy())

        cam = translation[i:i+1].squeeze(0) ###as 1,3##np.array([0.2, 0.42, -0.18]).reshape(1,3)#

        cam[:,0] = -cam[:,0]#####to s3d reference

        cam /= torch.max(d)

        d = d.reshape(rgb.shape[0], rgb.shape[1], 1) / torch.max(d)#####MOD no norm
        d_mask = (d<1e-3)
        ##d = torch.where( (d<1e-3), 1.0, d)
        d[d_mask] = 1
        
        coord = torch.cat((axis0, axis1, axis2), dim=2) * d

        ##img1, d1, _, _, mask1 = omninerf_translate_gpu(coord, rgb, d, cam)##FIXME

        img1 = omninerf_translate_gpu(coord, rgb, d, cam)
                
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
        mask1 = torch.where( (img1>0), 1.0, 0)###FIXMEmask1.unsqueeze(0).unsqueeze(0)
                
        tr_img.append(img1)
        tr_mask.append(mask1)

    tr_img = torch.cat(tr_img, dim=0)
    tr_mask = torch.cat(tr_mask, dim=0)

    return tr_img, tr_mask

    
    
if __name__ == '__main__':
    #from torch.utils.data import DataLoader
    #from data.structured3d import Structured3D, collate_fn
    #from torchvision.utils import save_image
    #from misc.utils import denorm, colorize

    #dataset = Structured3D('/p300/Structured3D', 'hard', 'train', 512)
    #loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    #image, target, camera1, camera2, depth, _ = next(iter(loader))
    #splatted = render(image, depth, camera2-camera1, 10)

    #save_image(denorm(image), 'vis/image.png')
    #save_image(denorm(target), 'vis/target.png')
    #save_image(denorm(splatted), 'vis/render.png')

    H,W = 512,1024
        
    device = torch.device('cuda') 

    h,w = 256,512##512,1024

    B = 2

    input = torch.randn(B,3,h,w).to(device)
    depth = torch.randn(B,h,w).to(device)
    x_t = torch.randn(B,1,3).to(device)

    img1,_ = render(input, depth, x_t)

    print(img1.shape)

