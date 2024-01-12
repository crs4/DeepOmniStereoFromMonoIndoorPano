import torch
import torch.nn.functional as F
import numpy as np


def coords2uv(coords):
    B, _, H, W = coords.size()
    u = ((coords[:, 0, :, :] + 0.5) / W - 0.5) * 2 * np.pi
    v = -((coords[:, 1, :, :] + 0.5) / H - 0.5) * np.pi
    uv = torch.stack([u, v], dim=1)
    return uv


def uv2xyzN(uv):
    xs = torch.cos(uv[:, 1, :, :]) * torch.sin(uv[:, 0, :, :])
    ys = torch.cos(uv[:, 1, :, :]) * torch.cos(uv[:, 0, :, :])
    zs = torch.sin(uv[:, 1, :, :])
    xyz = torch.stack([xs, ys, zs], dim=1)
    return xyz


def xyz2uv(xyz):
    x = xyz[:, 0, :, :]
    y = xyz[:, 1, :, :]
    z = xyz[:, 2, :, :]
    u = torch.atan2(x, y)
    v = torch.atan(z / torch.sqrt(x.pow(2) + y.pow(2)))
    uv = torch.stack([u, v], dim=1)
    return uv


def uv2coords(uv):
    _, _, H, W = uv.size()
    u = uv[:, 0, :, :]
    v = uv[:, 1, :, :]
    coordx = (u / (2 * np.pi) + 0.5) * W - 0.5
    coordy = (0.5 - v / np.pi) * H - 0.5
    coords = torch.stack([coordx, coordy], dim=1)
    return coords


def xyz2coords(xyz):
    _, _, H, W = xyz.size()
        
    xs = xyz[:, 0, :, :]
    ys = xyz[:, 1, :, :]
    zs = xyz[:, 2, :, :]
    us = torch.atan2(xs, ys)
    vs = torch.atan(zs / torch.sqrt(xs.pow(2) + ys.pow(2)))
    coordx = (us / (2 * np.pi) + 0.5) * W - 0.5
    coordy = (0.5 - vs / np.pi) * H - 0.5
    coords = torch.stack([coordx, coordy], dim=1)
        
    return coords

def xyz2depth(xyz):
    _, _, H, W = xyz.size()
        
    xs = xyz[:, 0, :, :]
    ys = xyz[:, 1, :, :]
    zs = xyz[:, 2, :, :]

    xx = xs * xs
    yy = ys * ys
    zz = zs * zs
        
    depth = torch.sqrt(xx+yy+zz)
        
    return depth


def transform_coords(depth, translation):
    N, _, H, W = depth.size()
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    coords = torch.stack([X, Y], dim=0).unsqueeze(0).repeat(N, 1, 1, 1).to(depth.device)
    uv = coords2uv(coords)
    xyz_unit = uv2xyzN(uv)
    xyz_camera = xyz_unit * depth
    xyz_camera_new = xyz_camera - translation.view(N, 3, 1, 1)
    coords_new = xyz2coords(xyz_camera_new)
    return coords_new

def transform_depthmap(depth, translation):
    N, _, H, W = depth.size()
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    coords = torch.stack([X, Y], dim=0).unsqueeze(0).repeat(N, 1, 1, 1).to(depth.device)
    uv = coords2uv(coords)
    xyz_unit = uv2xyzN(uv)
    xyz_camera = xyz_unit * depth
    xyz_camera_new = xyz_camera - translation.view(N, 3, 1, 1)

    coords_new = xyz2depth(xyz_camera_new).unsqueeze(1)
    
    return coords_new
