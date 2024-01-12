import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import os
import json
from collections import OrderedDict
from misc import config, post_proc, sobel

import torch.nn.functional as F

import math

def save_feature_map(feats, channel=1, id=512):
    import matplotlib.pyplot as plt
    from tqdm import trange

    if(feats.shape[3] == id):
       print('mask shape', feats.shape)
       
       max_iter = feats.shape[1]
       
       ch_id = 0
       
       for _ in trange(max_iter):
        fm = torch.softmax(feats,dim=1)[:, ch_id:ch_id+1].cpu().squeeze(1).squeeze(0).numpy()

        plt.figure(111+ch_id)
        ##plt.title('feats')
        plt.imshow(fm)

        ch_id += 1

def resize_crop(img, scale, size):
    
    re_size = int(img.shape[0]*scale)

    if(re_size>0):
        img = cv2.resize(img, (re_size, re_size), cv2.INTER_AREA)

    if size <= re_size:
        pd = int((re_size-size)/2)
        img = img[pd:pd+size,pd:pd+size]
    else:
        new = np.zeros((size,size))
        pd = int((size-re_size)/2)
        new[pd:pd+re_size,pd:pd+re_size] = img[:,:]
        img = new

    return img

def resize(img, scale):
    
    re_size = int(img.shape[0]*scale)
    if(re_size>0):
        img = cv2.resize(img, (re_size, re_size), cv2.INTER_CUBIC)
        
    return img

def var2np(data_lst):

    def trans(data):
        if data.shape[1] == 1:
            return data[0, 0].data.cpu().numpy()
        elif data.shape[1] == 3: 
            return data[0, :, :, :].permute(1, 2, 0).data.cpu().numpy()

    if isinstance(data_lst, list):
        np_lst = []
        for data in data_lst:
            np_lst.append(trans(data))
        return np_lst
    else:
        return trans(data_lst)

def save_map(img,name):
    
    vis = Image.fromarray(np.uint8(img* 255))
    vis.save(name)


def x2image(x):
    img = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
        
    return img

def recover_h_value(mask):
    return np.amax(mask.numpy())


def group_weight(module):
    # Group module parameters into two group
    # One need weight_decay and the other doesn't
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]

def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        scale_running_lr = max((1. - frac), 0.) ** args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr

def save_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.backbone,
            'full_size': net.full_size,
            'decoder_type': net.decoder_type,
            },
        'state_dict': net.state_dict(),
    })
    torch.save(state_dict, path)

def save_parallel_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.module.backbone,
            'full_size': net.module.full_size,
            'decoder_type': net.module.decoder_type,
            },
        'state_dict': net.module.state_dict()
    })
    torch.save(state_dict, path)

def save_emptying_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.backbone,
            'full_size': net.full_size,
            'decoder_type': net.decoder_type,
            },
        'state_dict': net.image_synth.state_dict(),
    })
    torch.save(state_dict, path)

def save_parallel_emptying_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.module.backbone,
            'full_size': net.module.full_size,
            'decoder_type': net.module.decoder_type,
            },
        'state_dict': net.module.image_synth.state_dict()
    })
    torch.save(state_dict, path)

def save_combo_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.backbone,
            'full_size': net.full_size,
            'decoder_type': net.decoder_type,
            },
        'state_dict': net.state_dict(),
    })
    torch.save(state_dict, path)

def save_parallel_combo_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.module.backbone,
            'full_size': net.module.full_size,
            'decoder_type': net.module.decoder_type,
            },
        'state_dict': net.module.state_dict()
    })
    torch.save(state_dict, path)

def load_combo_trained_model(Net, device, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(device, **state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net



def load_trained_model(Net, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net

###FIXMEEE
def load_synth_trained_model(Net, device, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(device, **state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net

def load_gated_trained_model(Net, device, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(device, **state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net

def load_emptying_room_trained_model(Net, device, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(device, **state_dict['kwargs'])
    net.image_synth.load_state_dict(state_dict['state_dict'])
    return net

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi

def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)

    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)

    coorys = v2coory(vs, h = h)
        
    return np.stack([coorxs, coorys], axis=-1)    


def createPointCloud(color, depth, ply_file):
    color = color.permute(1, 2, 0)
    print(color.shape)
    print(depth.shape)
    ### color:np.array (h, w)
    ### depth: np.array (h, w)

    pcSampleStride = 30

    heightScale = float(color.shape[0]) / depth.shape[0]
    widthScale = float(color.shape[1]) / depth.shape[1]

    points = []
    for i in range(color.shape[0]):
        if not i % pcSampleStride == 0:
            continue
        for j in range(color.shape[1]):
            if not j % pcSampleStride == 0:
                continue

            rgb = (color[i][j][0], color[i][j][1], color[i][j][2])

            d = depth[ int(i/heightScale) ][ int(j/widthScale) ]
            if d <= 0:
                continue

            coordsX = float(j) / color.shape[1]
            coordsY = float(i) / color.shape[0]

            xyz = coords2xyz((coordsX, coordsY) ,d)

            ##point = (xyz, rgb)
            ##pointCloud.append(point)

            points.append("%f %f %f %d %d %d 0\n"%(xyz[0],xyz[1],xyz[2],rgb[0],rgb[1],rgb[2]))

        file = open(ply_file,"w")
        file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        '''%(len(points),"".join(points)))
        file.close()

        
        #if i % int(color.shape[0]/10) == 0:
        #    print("PC generating {0}%".format(i/color.shape[0]*100))
    
    return points

def coords2xyz(coords, N):

    uv = coords2uv(coords)
    xyz = uv2xyz(uv, N)
    
    return xyz

def coords2uv(coords):  
    #coords: 0.0 - 1.0
    coords = (coords[0] - 0.5, coords[1] - 0.5)

    uv = (coords[0] * 2 * math.pi,
            -coords[1] * math.pi)

    return uv

def uv2xyz(uv, N):

    x = math.cos(uv[1]) * math.sin(uv[0])
    y = math.sin(uv[1])
    z = math.cos(uv[1]) * math.cos(uv[0])
    ##Flip Z　axis
    xyz = (N * x, N * y, -N * z)

    return xyz

def SphereGrid(equ_h, equ_w):
    cen_x = (equ_w - 1) / 2.0
    cen_y = (equ_h - 1) / 2.0
    theta = (2 * (np.arange(equ_w) - cen_x) / equ_w) * np.pi
    phi = (2 * (np.arange(equ_h) - cen_y) / equ_h) * (np.pi / 2)
    theta = np.tile(theta[None, :], [equ_h, 1])
    phi = np.tile(phi[None, :], [equ_w, 1]).T

    x = (np.cos(phi) * np.sin(theta)).reshape([equ_h, equ_w, 1])
    y = (np.sin(phi)).reshape([equ_h, equ_w, 1])
    z = (np.cos(phi) * np.cos(theta)).reshape([equ_h, equ_w, 1])
    xyz = np.concatenate([x, y, z], axis=-1)

    return xyz

def depth2pts(depth):
    grid = SphereGrid(*depth.shape) ### (h,w)
    pts = depth[..., None] * grid

    return pts

def image_depth_to_world(d):
    ##print('tools',d.shape)
    P = np.zeros(shape =(d.shape[0],d.shape[1],3),dtype=float)
    for i in range(d.shape[0]):
        theta = -np.pi * (float(i)/float(d.shape[0]-1)-0.5)
        for j in range(d.shape[1]):
            # check if there is replication
            phi = np.pi * (2.0*float(j)/float(d.shape[1]-1)-1.0) 
                                  
            P[i,j,0] = d[i,j]*math.cos(phi)*math.cos(theta)
            P[i,j,1] = d[i,j]*math.sin(theta)
            P[i,j,2] = d[i,j]*math.sin(phi)*math.cos(theta)
            
    return P

def depth_pixel_to_world(d, i, j):
    P = np.zeros(shape =(1, 1, 3),dtype=float)
       
    theta = -np.pi * (float(i)/float(d.shape[0]-1)-0.5)
    phi = np.pi * (2.0*float(j)/float(d.shape[1]-1)-1.0) 
    
    P[i,j,0] = d[i,j]*math.cos(phi)*math.cos(theta)
    P[i,j,1] = d[i,j]*math.sin(theta)
    P[i,j,2] = d[i,j]*math.sin(phi)*math.cos(theta)
            
    return P


def export_obj(outfile, P, rgb):
    rgb = 255.0 * rgb
    #P = 65535.0 * P 
    f = open(outfile, "w")
    f.write('# obj point cloud')
    for i in range (P.shape[0]):
        for j in range (P.shape[1]):  
            d  = P[i,j,0]**2 + P[i,j,1]**2 + P[i,j,2]**2
            if d > 1.0e-6:
                f.write('v %f %f %f %f %f %f \n'%(P[i,j,0],P[i,j,1],P[i,j,2],rgb[i,j,0],rgb[i,j,1],rgb[i,j,2]))
    f.close()

def export_model(img_depth,img_rgb,out_file):
    print('rgb',img_rgb.shape)
    print('depth',img_depth.shape)
    P = image_depth_to_world(img_depth)
    export_obj(out_file,P,img_rgb)

def export_from_batch(xyz_batch,img_rgb,out_file):
    print('rgb',img_rgb.shape)
    print('xyz', xyz_batch.shape)
    
    P = xyz_batch.squeeze(0)
    P = P.permute(1,2,0)

    export_obj(out_file, P, img_rgb)


def depth2normals(depth):
    ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float()
    
    get_gradient = sobel.Sobel()

    depth_grad = get_gradient(depth)
       
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    depth_normal = F.normalize(depth_normal, p=2, dim=1)
    
    return depth_normal





