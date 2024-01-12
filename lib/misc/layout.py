import torch
import torch.nn.functional as F
import torch.nn as nn

from misc.atlanta_transform import E2P
from misc import epc, tools, panorama, post_proc

import cv2
import numpy as np
import math
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import LineString
#from scipy.spatial.distance import cdist
from shapely.geometry import Polygon

##from skimage.feature import canny

###DEBUG
import matplotlib.pyplot as plt

PI = float(np.pi)


def load_layout_from_txt(cor, H, W):
    # Corner with minimum x should at the beginning
    cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)
        
    # Prepare 1d ceiling-wall/floor-wall boundary
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)

    for i in range(n_cor // 2):
        xys = tools.pano_connect_points(cor[i*2],
                                                cor[(i*2+2) % n_cor],
                                                z=-50)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])

    for i in range(n_cor // 2):
        xys = tools.pano_connect_points(cor[i*2+1],
                                                cor[(i*2+3) % n_cor],
                                                z=50)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])

    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)

    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)

    ##bon = ((bon + 0.5) / H - 0.5) * np.pi
    ##e_bon = (e_bon / np.pi + 0.5) * H - 0.5
   
   
    #cv2.fillPoly(mask, [c_pts], 255)

    return bon
     
def visualize_panorama(cor_id, img_src, pid = 123, name = 'visualize equi layout'):
    ####from torh batch BxNx2
             
    np_cor_id = cor_id.numpy()

    # Detect occlusion
    ##occlusion = find_occlusion(np_cor_id[::2].copy()).repeat(2)    
    ##np_cor_id = np_cor_id[~occlusion]

    img_viz = panorama.draw_boundary_from_cor_id(np_cor_id, img_src)

    plt.figure(pid)
    plt.title(name)
    ##plt.axis('off')
    plt.imshow(img_viz)

def get_layout_edges(cor_id, im_h, im_w):
    ####from 1xNx2
    np_cor_id = cor_id.squeeze(0).numpy()
        
    cor_all = [np_cor_id]

    for i in range(len(np_cor_id)):
        cor_all.append(np_cor_id[i, :])
        cor_all.append(np_cor_id[(i+2) % len(np_cor_id), :])
    
    cor_all = np.vstack(cor_all)

    rs, cs = panorama.lineIdxFromCors(cor_all, im_w, im_h)
    rs = np.array(rs)
    cs = np.array(cs)

    panoEdge = np.zeros([im_h, im_w], np.uint8)

    for dx, dy in [[-1, 0], [1, 0], [0, 0], [0, 1], [0, -1]]:
        ##panoEdgeC[np.clip(rs + dx, 0, im_h - 1), np.clip(cs + dy, 0, im_w - 1), 0] = 0
        ##panoEdgeC[np.clip(rs + dx, 0, im_h - 1), np.clip(cs + dy, 0, im_w - 1), 1] = 0
        panoEdge[np.clip(rs + dx, 0, im_h - 1), np.clip(cs + dy, 0, im_w - 1)] = 1

    return panoEdge

def batched_layout_edges(b_cor_id, im_h, im_w):
    batch_size = b_cor_id.size()[0]
        
    LE = []

    for i in range(batch_size):
        np_le = get_layout_edges(b_cor_id[i:i+1], im_h, im_w)
                        
        LE.append(torch.FloatTensor(np_le).unsqueeze(0))####to batch shape 1xNx2

    LE = torch.cat(LE, dim=0)####BxNx2

    return LE 

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys

def find_occlusion(coor, w = 1024, h = 512):
    u = tools.coorx2u(coor[:, 0], w)
    v = tools.coory2v(coor[:, 1], h)
    x, y = tools.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i+1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)

def cor_2_1d(cor, H, W, to_angles = False): #####return ceiling and floor boundaries as a 1D signal along W
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []

    n_cor = len(cor)
    
    for i in range(n_cor // 2):
        xys = tools.pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])

    for i in range(n_cor // 2):
        xys = tools.pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])

    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)

    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)

    ##print(bon[1])

    if(to_angles):
        bon = ((bon + 0.5) / H - 0.5) * np.pi
        
    ###test
    #for j in range(len(bon[0])):
    #    now_x = bon[0, j]
    #    now_y = bon[1, j]
    #    print(now_x,now_y)

    return bon

def layout_2_depth(cor_id, h, w, min, return_mask=False, get_depth_edges = False, filter_iter = 1):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
            
    vc, vf = cor_2_1d(cor_id, h, w, to_angles = True)
                        
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
        
    ##FIXMEassert (vc > 0).sum() == 0
    ##FIXMEassert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]
        
    # Floor-plane to depth
    floor_h = min
    floor_d = np.abs(floor_h / np.sin(vs))

    ##print('layout h',floor_h)

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    
    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]

    ##print('layout_2_depth h',floor_h, ceil_h)

    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
       
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    ##assert (depth == 0).sum() == 0       
        

    if(get_depth_edges):
        vci, vfi = cor_2_1d(cor_id, h, w, to_angles = False)               
       
        vci = vci[None, :]  # [1, w]
        vfi = vfi[None, :]  # [1, w]

        ##vsi = np.arange(h)##((np.arange(h) + 0.5) / h - 0.5) * np.pi
        ##vsi = np.repeat(vsi[:, None], w, axis=1)  # [h, w]

        vx = np.arange(w)
        vx = vx[None, :]  # [1, w]

        ####cat coords

        vc_coords = np.concatenate((vx,vci),axis=0)
        vf_coords = np.concatenate((vx,vfi),axis=0)

        vc_coords = np.transpose(vc_coords,(1,0)).astype(int)
        vf_coords = np.transpose(vf_coords,(1,0)).astype(int)

        b_coords = np.concatenate((vc_coords, vf_coords),axis=0)
                
        cont_mask = np.zeros(shape=(h, w), dtype=np.uint8)
        
        ##print('b_coords',b_coords.shape,'cont_mask',cont_mask.shape)

        ##print(b_coords[:, 1], b_coords[:, 0])
        
        cont_mask[b_coords[:, 1], b_coords[:, 0]] = 1
               
        # Detect occlusion
        np_cor_id = cor_id

        ##print(np_cor_id)

        occlusion = find_occlusion(np_cor_id[::2].copy(), w = w, h = h).repeat(2)    
        np_cor_id = np_cor_id[~occlusion]

        ##print('occ',np_cor_id)
                
        ###TO DO draw vertical lines
        for i in range(len(np_cor_id)//2):
            p1 = np_cor_id[i*2].astype(int)
            p2 = np_cor_id[i*2+1].astype(int)

            x0 = p1[0]

            y1 = p1[1]
            y2 = p2[1]
                                            
            l = np.linspace(p1,p2,(y2-y1), retstep=True, dtype=int,axis=1)

            v_edge = np.transpose(l[0],(1,0)).astype(int)

            cont_mask[v_edge[:, 1], v_edge[:, 0]] = 1
       
        #plt.figure(456)
        #plt.title('DEBUG layout depth')
        #plt.imshow(cont_mask) 

        ## make edges in bold
        
        cont_mask = torch.FloatTensor(cont_mask).unsqueeze(0).unsqueeze(0)
        if(filter_iter>0):           
            #
            for i in range(filter_iter):        
                cont_mask = F.interpolate(cont_mask, size=(h//2, w//2), mode='bilinear', align_corners=False)
                cont_mask = F.interpolate(cont_mask, size=(h, w), mode='bilinear', align_corners=False)

        cont_mask = torch.where(cont_mask > 0.0, 1.0, 0.)
        cont_mask = cont_mask.squeeze(0).squeeze(0).numpy()


        return depth, cont_mask

    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask

    return depth

def layout_2_segmentation(cor_id, h, w, get_edges = False, as_layers = False, filter_iter = 1):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2    
    vc, vf = cor_2_1d(cor_id, h, w, to_angles = True)
                        
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]        
    ##FIXMEassert (vc > 0).sum() == 0
    ##FIXMEassert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]
      
   
    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
       
    ####all masks in a single channel
    seg_mask = np.zeros([h, w], np.float32)    # [h, w]
    seg_mask[floor_mask] = 64.0##floor_d[floor_mask]
    seg_mask[ceil_mask]  = 128.0##ceil_d[ceil_mask]
    seg_mask[wall_mask]  = 255.0##wall_d[wall_mask]

    seg_mask = torch.FloatTensor(seg_mask).unsqueeze(0)

    ##assert (depth == 0).sum() == 0  

    if(as_layers):
            SM = []

            SM.append(torch.FloatTensor(floor_mask).unsqueeze(0))
            SM.append(torch.FloatTensor(wall_mask).unsqueeze(0))
            SM.append(torch.FloatTensor(ceil_mask).unsqueeze(0))

            seg_mask = torch.cat(SM, dim=0) 
           

    if(get_edges):
        vci, vfi = cor_2_1d(cor_id, h, w, to_angles = False)               
       
        vci = vci[None, :]  # [1, w]
        vfi = vfi[None, :]  # [1, w]

        ##vsi = np.arange(h)##((np.arange(h) + 0.5) / h - 0.5) * np.pi
        ##vsi = np.repeat(vsi[:, None], w, axis=1)  # [h, w]

        vx = np.arange(w)
        vx = vx[None, :]  # [1, w]

        ####cat coords

        vc_coords = np.concatenate((vx,vci),axis=0)
        vf_coords = np.concatenate((vx,vfi),axis=0)

        vc_coords = np.transpose(vc_coords,(1,0)).astype(int)
        vf_coords = np.transpose(vf_coords,(1,0)).astype(int)

        b_coords = np.concatenate((vc_coords, vf_coords),axis=0)

        cont_mask = np.zeros(shape=(h, w), dtype=np.uint8)
        cont_mask[b_coords[:, 1], b_coords[:, 0]] = 1
               
        # Detect occlusion
        np_cor_id = cor_id
        occlusion = find_occlusion(np_cor_id[::2].copy(), w = w, h = h).repeat(2)    
        np_cor_id = np_cor_id[~occlusion]
                
        ###TO DO draw vertical lines
        for i in range(len(np_cor_id)//2):
            p1 = np_cor_id[i*2].astype(int)
            p2 = np_cor_id[i*2+1].astype(int)

            x0 = p1[0]

            y1 = p1[1]
            y2 = p2[1]
                    
            l = np.linspace(p1,p2,(y2-y1), retstep=True, dtype=int,axis=1)
                        
            v_edge = np.transpose(l[0],(1,0)).astype(int)

            cont_mask[v_edge[:, 1], v_edge[:, 0]] = 1
       
        #plt.figure(456)
        #plt.title('DEBUG layout depth')
        #plt.imshow(cont_mask) 

        ## make edges in bold
                        
        if(filter_iter>0):
            cont_mask = torch.FloatTensor(cont_mask).unsqueeze(0).unsqueeze(0)

            for i in range(filter_iter):        
                cont_mask = F.interpolate(cont_mask, size=(h//2, w//2), mode='bilinear', align_corners=False)
                cont_mask = F.interpolate(cont_mask, size=(h, w), mode='bilinear', align_corners=False)

            cont_mask = torch.where(cont_mask > 0.0, 1.0, 0.)
            cont_mask = cont_mask.squeeze(0) ####1xhxw
            
        
        seg_mask = torch.cat((seg_mask,cont_mask), dim=0)
        
           
    return seg_mask
  
def MW_post_processing(xs_,y_bon_,W, H, z0, z1, post_force_cuboid = False):
    min_v = 0 if post_force_cuboid else 0.05
    r = int(round(W * 0.05 / 2))
    N = 4 if post_force_cuboid else None
        
    # Generate wall-walls - using xs_ corners (peaks) and ceiling boundary y_bon_[0]
    cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=post_force_cuboid)#####TO DO check gen_ww

    ##print('MW temp results',cor.shape)

    ###DEBUG
    xy = post_proc.np_coor2xy(cor, z0, W, H, floorW=0, floorH=0) #####XY cartesian coords
    ##print(xy)
    ##xy_c = 
    
    if not post_force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)

        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
        
        ####failed to generate: use cuboid
        if not Polygon(xy2d).is_valid:
            #import sys
            #print(
            #    'Fail to generate valid general layout!! '
            #    'Generate cuboid as fallback.',
            #    file=sys.stderr)
            ##xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]####TO DO remove
            cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    ##print('post proc cor_id',cor_id)

    return cor_id, xy
    
def xy2coor(xy, z=50, coorW=1024, coorH=512):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] ##- floorW / 2 + 0.5
    y = xy[:, 1] ##- floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])

def batched_transform_equi_corners(e_pts, max_min, H, W, tc_batch, theta_sort = False):
    batch_size = e_pts.size()[0]
        
    TE = []

    for i in range(batch_size):
        max = max_min[i:i+1].squeeze(0)[0]
        min = max_min[i:i+1].squeeze(0)[1]

        max = max.cpu().numpy()
        min = min.cpu().numpy()
               
        tc = tc_batch[i:i+1].squeeze(0).cpu().numpy()
                               
        np_le = transform_equi_corners(e_pts[i:i+1].squeeze(0).cpu().numpy(), max, min, H, W, tc)
                                        
        TE.append(torch.FloatTensor(np_le).unsqueeze(0))####to batch shape 1xNx2

    TE = torch.cat(TE, dim=0)####BxNx2

    return TE

def transform_equi_corners(e_pts, c_dist, f_dist, H, W, translation = np.zeros((1,3)), theta_sort = False): 
        ###input (N,2)
                                                                   
        ###spitting ceiling and floor         
        c_pts_x = []
        c_pts_y = []

        n_cor = len(e_pts)
        for i in range(n_cor // 2):
            c_pts_x.append(e_pts[i*2][0])
            c_pts_y.append(e_pts[i*2][1])
                        
        c_pts_x = np.array(c_pts_x)
        c_pts_y = np.array(c_pts_y)

        c_pts = np.stack((c_pts_x, c_pts_y), axis=-1)#########ceiling equi corners (N,2)
        #################################################

        ####convert to cartesian XY
        XY = post_proc.np_coor2xy(c_pts, c_dist, W, H, floorW=1.0, floorH=1.0)
                        
        #####translate
        x_tr = translation[:,:1]
        y_tr = translation[:,1:2]
        z_tr  = translation[:,2:] 
        
        XY[:,:1]  -= x_tr
        XY[:,1:2] += y_tr

        off = (z_tr.squeeze(0))[0]
                
        c_dist -= off              
        f_dist += off

            
        ####create translated equi coords
        ####default: cartesian order
        tr_c_cor = xy2coor(np.array(XY),  c_dist, W, H)
        tr_f_cor = xy2coor(np.array(XY), -f_dist, W, H) ####based on the ceiling shape
                
        cor_count = len(tr_c_cor)                                                      
                               
        if(theta_sort):
            ####sorted by theta (pixels coords)
            c_ind = np.lexsort((tr_c_cor[:,1],c_cor[:,0])) 
            f_ind = np.lexsort((tr_f_cor[:,1],f_cor[:,0]))
            tr_c_cor = tr_c_cor[c_ind]
            tr_f_cor = tr_f_cor[f_ind]
                       
        tr_equi_coords = []
                
        for j in range(len(tr_c_cor)):
            tr_equi_coords.append(tr_c_cor[j])
            tr_equi_coords.append(tr_f_cor[j])
               
        tr_e_pts = np.array(tr_equi_coords)

        return tr_e_pts


class D2L(nn.Module):
    def __init__(self, gpu=False, H = 512, W = 1024, fp_size = 512, fp_fov = 165.0):
        super(D2L, self).__init__()

        self.fp_size = fp_size
        self.fp_fov = fp_fov

        self.img_size = [W,H]

        self.ep = epc.EPC(gpu=gpu)
        self.e2p = E2P(equ_size=(H, W), out_dim=self.fp_size, fov=self.fp_fov, radius=1, gpu = gpu, return_fl = True)

        self.xz_sph = self.ep.atlanta_sphere(H, W)

    def get_segmentation_masks(self, seg_pred):
        soft_sem = torch.softmax(seg_pred, dim = 1) #####TO DO - here semantic is given by clutter mask
        soft_sem = torch.argmax(soft_sem, dim=1, keepdim=True)
        soft_sem = torch.clamp(soft_sem, min=0, max=1)
        masks = torch.zeros_like(seg_pred).to(seg_pred.device)
        masks.scatter_(1, soft_sem, 1)
                        
        return masks

    def get_src_layout(self, src_depth, src_layout_seg):
        mask_pred = self.get_segmentation_masks(src_layout_seg)
        layout_mask = mask_pred[:,:1]##

        B,H,W = src_depth.size()
        
        LS = []

        for i in range(B):
            ##print('element batch shape', src_depth[i:i+1].shape, self.xz_sph.shape)
            p_max, p_min = self.max_min_depth(src_depth[i:i+1])

            #p_max = p_max.cpu().numpy()
            #p_min = p_min.cpu().numpy()
                        
            m_contour = self.contour_from_cmask(layout_mask.cpu(), epsilon_b=0.01)####return numpy contour

            LS.append(m_contour)

        ##src_layout = np.concatenate(LS, axis=0)##.to(src_depth.device)####Bx1xhxw
                        
    
        return LS, p_max, p_min

    def translated_edges_from_layout(self, src_layout, p_max, p_min, W, H, x_c):####from numpy arrays
        B = len(src_layout)
                        
        LE = []
                
        for i in range(B):
            equi_pts, equi_1D, cart_XY = self.contour_pts2equi_layout(src_layout[i], W, H, p_max, p_min, translation = x_c)

            le = get_layout_edges(torch.FloatTensor(equi_pts).unsqueeze(0), H, W)

            le = torch.FloatTensor(le).unsqueeze(0).unsqueeze(0)

            LE.append(le)

        layout_edges = torch.cat(LE, dim=0)####Bx1xhxw

        return layout_edges

    
    def get_translated_layout_edges(self, src_depth, src_layout_seg, x_c):#######batched
        
        mask_pred = self.get_segmentation_masks(src_layout_seg)
        layout_mask = mask_pred[:,:1]##

        B,H,W = src_depth.size()
        
        LE = []

        for i in range(B):
            ##print('element batch shape', src_depth[i:i+1].shape, self.xz_sph.shape)
            p_max, p_min = self.max_min_depth(src_depth[i:i+1])

            p_max = p_max.cpu().numpy()
            p_min = p_min.cpu().numpy()
                        
            m_contour = self.contour_from_cmask(layout_mask.cpu(), epsilon_b=0.01)####return numpy contour
                                                
            ##equi_pts, equi_1D, cart_XY = dataset.d2l.contour_pts2equi_layout(m_contour, x_img.shape[3], x_img.shape[2], p_max, p_min, theta_sort = False)
            equi_pts, equi_1D, cart_XY = self.contour_pts2equi_layout(m_contour, W, H, p_max, p_min, translation = x_c.squeeze(0).numpy())

            le = get_layout_edges(torch.FloatTensor(equi_pts).unsqueeze(0), H, W)

            le = torch.FloatTensor(le).unsqueeze(0).unsqueeze(0)

            LE.append(le)

        layout_edges = torch.cat(LE, dim=0)####Bx1xhxw
                        
    
        return layout_edges
    
    ################from c-f bondaries returns (eventually translated): layout atlanta mask, xy coords, max, min, e_pts  
    def atlanta_transform_from_equi_corners(self, e_pts, max, min, translation = np.zeros((1,3)), theta_sort = False): 
                                                           
        ###spitting ceiling and floor
        W = self.img_size[0]
        H = self.img_size[1]  
                       
        c_pts_x = []
        c_pts_y = []

        n_cor = len(e_pts)
        for i in range(n_cor // 2):
            c_pts_x.append(e_pts[i*2][0])
            c_pts_y.append(e_pts[i*2][1])
                        
        c_pts_x = np.array(c_pts_x)
        c_pts_y = np.array(c_pts_y)

        c_pts = np.stack((c_pts_x, c_pts_y), axis=-1)#########ceiling equi corners (N,2)
        #################################################

        ####convert to cartesian XY
        XY = post_proc.np_coor2xy(c_pts, max, W, H, floorW=1.0, floorH=1.0)
        
        #####translate
        x_tr = translation[:,:1]
        y_tr = translation[:,1:2]
        z_tr  = translation[:,2:]               
        XY[:,:1]  -= x_tr
        XY[:,1:2] += y_tr
        max -= z_tr.squeeze(0)
        min += z_tr.squeeze(0)
                
        #post_poly = Polygon(XY)
        #plt.figure(142)
        #plt.title('atlanta from txt')
        #plt.gca().invert_yaxis()    
        #plt.axes().set_aspect('equal')
        #plt.plot(*post_poly.exterior.xy,color='green')
        #
        
        ####create translated equi coords
        ####default: cartesian order
        tr_c_cor = xy2coor(np.array(XY),  max, W, H)
        tr_f_cor = xy2coor(np.array(XY), -min, W, H) ####based on the ceiling shape
                
        cor_count = len(tr_c_cor)                                                      
                               
        if(theta_sort):
            ####sorted by theta (pixels coords)
            c_ind = np.lexsort((tr_c_cor[:,1],c_cor[:,0])) 
            f_ind = np.lexsort((tr_f_cor[:,1],f_cor[:,0]))
            tr_c_cor = tr_c_cor[c_ind]
            tr_f_cor = tr_f_cor[f_ind]
                       
        tr_equi_coords = []
                
        for j in range(len(tr_c_cor)):
            tr_equi_coords.append(tr_c_cor[j])
            tr_equi_coords.append(tr_f_cor[j])
               
        tr_e_pts = np.array(tr_equi_coords)
        
        
        ###world to atlanta ratio
        h_ratio = min / max
        fp_meter = min / math.tan(math.pi *  (180.0 - self.fp_fov) / 360.0) * 2.0
        e2p_px_size = (fp_meter / self.fp_size) ##* 100.0
        c_scale = (1.0/e2p_px_size)
        ###############################         
        
        #####convert translated equi corners to scaled atlanta xy #####FIXMEEEE replace with translated coords
        atl_xy = post_proc.np_coor2xy(tr_c_cor, max, W, H, floorW=self.fp_size, floorH=self.fp_size, m_ratio = h_ratio*c_scale)
        ###create atlanta mask
        mask = np.zeros([self.fp_size, self.fp_size], np.uint8) 
        ##mask = np.zeros([self.fp_size, self.fp_size], np.single)                   
        m_pts = atl_xy.astype(np.int32)
        m_pts = m_pts.reshape((-1,1,2))
        cv2.fillPoly(mask, [m_pts], 1)                                 
                   

        return torch.FloatTensor(mask), torch.FloatTensor(XY), max, min, torch.FloatTensor(tr_e_pts)
            
    def max_min_depth(self, src_depth):
        src_depth_plan = self.ep.euclidean_to_planar_depth(src_depth.squeeze(0), self.xz_sph).unsqueeze(0)##

        ##print(src_depth_plan.shape)

        B,_,H,W = src_depth_plan.size()

        up_depth, bottom_depth = torch.split(src_depth_plan, H//2, dim=2)

        max = torch.max(up_depth)
        min = torch.max(bottom_depth)
               
        ##print(max, min)

        return max,min

    def convert_depth_mapping(self, src_depth, sphere_type = 'polar'):
        if(sphere_type == 'atlanta'):
            xz_sph = self.ep.atlanta_sphere(self.img_size[1], self.img_size[0])

        if(sphere_type == 'polar'):
            xz_sph = self.ep.polar_sphere(self.img_size[1], self.img_size[0])

        if(sphere_type == 'euclidean'):
            xz_sph = self.ep.xyz_sphere(self.img_size[1], self.img_size[0])
        
        src_depth_plan = self.ep.euclidean_to_planar_depth(src_depth.squeeze(0), xz_sph).unsqueeze(0)##

        return src_depth_plan


    
    def atlanta_transform_from_depth(self, src_depth):
        ####input depth 1xhxw - no batched
        src_depth_plan = self.ep.euclidean_to_planar_depth(src_depth.squeeze(0), self.xz_sph).unsqueeze(0)## input: ###### (hxw) - (1xhxw)
                                        
        [d_up, d_down, fl] = self.e2p(src_depth_plan) ####batched trans - same fl

        c_dist = torch.max(d_up)
        f_dist = torch.max(d_down)

        debug = False

        if(debug):
            plt.figure(1030)
            plt.title(' up depth src')
            plt.imshow(d_up.squeeze(0).squeeze(0)) 

            plt.figure(1031)
            plt.title(' down depth src')
            plt.imshow(d_down.squeeze(0).squeeze(0)) 

            plt.figure(1032)
            plt.title(' plan depth')
            plt.imshow(src_depth_plan.squeeze(0).squeeze(0)) 

           
        return d_up, d_down, c_dist, f_dist, fl

    def batched_atlanta_transform_from_depth(self, src_depth):
        ####input depth Bx1hxw - batched
        batch_size = src_depth.size()[0]
        
        DP = []

        for i in range(batch_size):
            ##print('element batch shape', src_depth[i:i+1].shape, self.xz_sph.shape)
            dp = self.ep.euclidean_to_planar_depth(src_depth[i:i+1], self.xz_sph) ## input: ###### (1xhxw) - (1xhxw)
            DP.append(dp)

        src_depth_plan = torch.cat(DP, dim=0)####Bx1xhxw
                
        ##src_depth_plan = self.ep.euclidean_to_planar_depth(src_depth.squeeze(0), self.xz_sph).unsqueeze(0)## input: 1xhxw , 1xwxh
                        
        [d_up, d_down, fl] = self.e2p(src_depth_plan) ####batched trans - same fl

        c_dist = torch.max(d_up)
        f_dist = torch.max(d_down)

        debug = False

        if(debug):
            plt.figure(1030)
            plt.title(' up depth src')
            plt.imshow(d_up.squeeze(0).squeeze(0)) 

            plt.figure(1031)
            plt.title(' down depth src')
            plt.imshow(d_down.squeeze(0).squeeze(0)) 

            plt.figure(1032)
            plt.title(' plan depth')
            plt.imshow(src_depth_plan.squeeze(0).squeeze(0)) 

           
        return d_up, d_down, c_dist, f_dist, fl
    
        
    def cmask_from_depth(self, src_depth):
        
        d_up, d_down, c_dist, f_dist, fl = self.atlanta_transform_from_depth(src_depth)                          
        
        c_th = c_dist * 0.95

        cmask = (d_up > c_th).float()                

        debug = False

        if(debug):
            
            plt.figure(1033)
            plt.title(' plan depth max')
            plt.imshow(cmask.squeeze(0).squeeze(0))

        return cmask, c_dist, f_dist, fl

    def atl_pts2xy(self, fp_pts, c_dist, f_dist):
        ###fp_pts.shape (N,1,2)
                
        fp_meter = (f_dist / math.tan(math.pi *  (180 - self.fp_fov) / 360) * 2)
               
        scale = c_dist / f_dist

        ##fp_pts = torch.Tensor(fp_pts).to(c_dist.device)###HACK
        fp_pts = fp_pts.astype(float)#######numpy contour
        fp_pts -= self.fp_size / 2.0
        fp_pts *= scale
        ##FIXfp_pts = fp_pts.astype(int)

        fp_meter /= self.fp_size

        
                               
        for i in range(fp_pts.shape[1]):
            fp_xy = fp_pts[:,i] * fp_meter
        
        ##fp_xy = fp_pts * (fp_meter / self.fp_size)
                        
        return fp_xy##.squeeze(1)

    ###########valid method from here
    def contour_pts2equi_layout( self, c_pts, W, H, c_dist, f_dist, translation = np.zeros((1,3)) ):####NB. use theta sort to store 2D footprint
        ##c_pts = c_pts.squeeze(1) 

        xy = self.atl_pts2xy(c_pts, c_dist, f_dist)
                        
        ####apply translation
        ##print(xy.shape, translation.shape)

        x_tr = translation[:,:1]
        y_tr = translation[:,1:2]
        z_tr  = translation[:,2:]

        ##print(xy_tr, z_tr)

        ##print('tr input',xy)

        xy[:,:1]  -= x_tr
        xy[:,1:2] += y_tr

        off = (z_tr.squeeze(0))[0]
                
        c_dist -= off              
        f_dist += off

        ##print('tr output',xy)
        
        ####default: cartesian order
        c_cor = xy2coor(np.array(xy),  c_dist, W, H)
        f_cor = xy2coor(np.array(xy), -f_dist, W, H) ####based on the ceiling shape

        cor_count = len(c_cor)                                                      
                           
                              
        equi_coords = []
                
        for j in range(len(c_cor)):
            equi_coords.append(c_cor[j])
            equi_coords.append(f_cor[j])
               
        equi_coords = np.array(equi_coords)
        xs = np.array(c_cor) 

        u,v = torch.unbind(torch.from_numpy(xs), dim=1)
                    
        return equi_coords, u.numpy().astype(np.uint32), xy
       

    def contour_from_cmask(self, mask, epsilon_b=0.005, get_valid = False):
        
        data_cnt, data_heri  = cv2.findContours(np.uint8(mask.squeeze(0).squeeze(0)), 1, 2)##CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE
        ##data_cnt, data_heri = cv2.findContours(data_thresh, 0, 2)##CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE
        
        is_valid = False
        
        approx = np.empty([1, 1, 2])
        
        if(len(data_cnt)>0):
            c = max(data_cnt, key = cv2.contourArea)
            epsilon = epsilon_b*cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            is_valid = True
        else:
            print('WARNING: no contour found')   
        
        # draw_mask = np.uint8(mask.squeeze(0).squeeze(0))
        # cv2.polylines(approx, [approx], True, (255),2,cv2.LINE_AA)
        # plt.figure(1034)
        # plt.title('contour')
        # plt.imshow(draw_mask)
        # #################################################

            
        if(get_valid):                
            return approx, is_valid
        else:
            return approx


    def forward(self, depth, tr = None):                   
        ###
        mask, max, min, fl = self.cmask_from_depth(depth)
        print('max', max, 'min', min, 'height', max+min, 'focal l', fl)

        m_contour = self.contour_from_cmask(mask)

        #draw_mask = np.uint8(mask.squeeze(0).squeeze(0))                        
        #cv2.polylines(draw_mask, [m_contour], True, (255),2,cv2.LINE_AA)

        #plt.figure(1034)
        #plt.title('contour')
        #plt.imshow(draw_mask)

        tr_in = np.zeros((1,3))

        if(tr is not None):
            tr_in = tr.numpy()

        use_post_proc = True

        if(use_post_proc):
            equi_pts, equi_1D, cart_XY = self.contour_pts2equi_layout(m_contour, self.img_size[0], self.img_size[1], max.numpy(), min.numpy(), translation = tr_in)
                                
            #pre_poly = Polygon(cart_XY)####NB polygon convert xy coords to image coords (1024x1024)
                           
                                        
            #plt.figure(141)
            #plt.title('XY pre')
            #plt.gca().invert_yaxis()    
            #plt.axes().set_aspect('equal')
            #plt.plot(*pre_poly.exterior.xy,color='red',alpha=0.8)
                        
                               
            ###using post_processing

            print(equi_pts.shape)

            y_bon_ = cor_2_1d(equi_pts, self.img_size[1], self.img_size[0])

            equi_pts, xy_cor = MW_post_processing(equi_1D, y_bon_, self.img_size[0], self.img_size[1], max.numpy(), -min.numpy(), post_force_cuboid = False)

            #post_poly = Polygon(xy_cor)
            #plt.figure(142)
            #plt.title('XY post')
            #plt.gca().invert_yaxis()    
            #plt.axes().set_aspect('equal')
            #plt.plot(*post_poly.exterior.xy,color='green')
                       
        else:               
            equi_pts, equi_1D, cart_XY = self.contour_pts2equi_layout(m_contour, self.img_size[0], self.img_size[1], max.numpy(), min.numpy(), translation = tr_in)  
                                                        
        ##print('post proc input',xs,y_bon_)            
               
        ##DL = torch.cat(DL, dim=0) ####to torch batch

        return torch.FloatTensor(equi_pts),max,min 