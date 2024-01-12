import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

import torch.nn.functional as F


import math

class EPC(nn.Module):
    def __init__(self, gpu=False, YZ_swap = True):
        super(EPC, self).__init__()

        self.gpu = gpu
                
        for param in self.parameters():
            param.requires_grad = False

        self.EPC = []

        self.YZ_swap = YZ_swap
                

    def xyz_sphere(self,H,W):####NB. phi and theta have different convention 
        ####build xyz sphere coordinates
        P = np.zeros(shape =(3, H, W),dtype=np.float32) 

        for i in range(H):
            theta = -np.pi * (float(i)/float(H-1)-0.5)
            for j in range(W):
                phi = np.pi * (2.0*float(j)/float(W-1)-1.0)
               
                if(self.YZ_swap):
                    P[0,i,j] = math.cos(phi)*math.cos(theta)
                    P[1,i,j] = math.sin(theta) ####Z
                    P[2,i,j] = math.sin(phi)*math.cos(theta)
                else:                
                    P[0,i,j] = math.cos(phi)*math.cos(theta)
                    P[1,i,j] = math.sin(phi)*math.cos(theta)
                    P[2,i,j] = math.sin(theta) ####Z

                        #
        if self.gpu:
            P = Variable(torch.FloatTensor(P)).cuda()
        else:
            P = Variable(torch.FloatTensor(P))

        P = P.unsqueeze(0) ###to batched shape ##xyz: 1x3xhxw
                    
        return P

    def polar_sphere(self,H,W):####NB. phi and theta have different convention
        ####build xyz sphere coordinates
        P = np.zeros(shape =(2, H, W),dtype=np.float32) 

        for i in range(H):
            theta = -np.pi * (float(i)/float(H-1)-0.5)
            for j in range(W):
                phi = np.pi * (2.0*float(j)/float(W-1)-1.0)
               
                P[0,i,j] = math.cos(phi)*math.cos(theta)
                ##P[1,i,j] = math.sin(theta) ####Z
                P[1,i,j] = math.sin(phi)*math.cos(theta)

        if self.gpu:
            P = Variable(torch.FloatTensor(P)).cuda()
        else:
            P = Variable(torch.FloatTensor(P))
        
        xx = P[0] * P[0]
        zz = P[1] * P[1]
        
        D = torch.sqrt(xx+zz)
                
        D = D.unsqueeze(0) ###to batched shape ##xyz: 1x3xhxw
                    
        return D

    def atlanta_sphere(self,H,W):#####NB. atlanta_plan - ####NB. phi and theta have different convention: phi is azimuth
        ####build xyz sphere coordinates
        P = np.zeros(shape =(3, H, W),dtype=np.float32) 

        for i in range(H):
            theta = -np.pi * (float(i)/float(H-1)-0.5)
            for j in range(W):
                phi = np.pi * (2.0*float(j)/float(W-1)-1.0)
               
                P[0,i,j] = 0.0#math.cos(phi)*math.cos(theta)
                P[1,i,j] = math.sin(theta) ####Z
                P[2,i,j] = 0.0#math.sin(phi)*math.cos(theta)

        if self.gpu:
            P = Variable(torch.FloatTensor(P)).cuda()
        else:
            P = Variable(torch.FloatTensor(P))
        
        xx = P[0] * P[0]
        zz = P[1] * P[1]
        yy = P[2] * P[2]
        
        D = torch.sqrt(xx+yy+zz)
                
        D = D.unsqueeze(0) ###to batched shape ##xyz: 1x3xhxw
                    
        return D

    def custom_sphere(self,H,W):#####NB. atlanta_plan - ####NB. phi and theta have different convention: phi is azimuth
        ####build xyz sphere coordinates
        P = np.zeros(shape =(3, H, W),dtype=np.float32) 

        for i in range(H):
            theta = -np.pi * (float(i)/float(H-1)-0.5)
            for j in range(W):
                phi = np.pi * (2.0*float(j)/float(W-1)-1.0)
               
                P[0,i,j] = math.cos(phi)#math.cos(theta)
                P[1,i,j] = 0.0##math.sin(theta) ####Z
                P[2,i,j] = math.sin(phi)##*math.cos(theta)

        if self.gpu:
            P = Variable(torch.FloatTensor(P)).cuda()
        else:
            P = Variable(torch.FloatTensor(P))
        
        xx = P[0] * P[0]
        zz = P[1] * P[1]
        yy = P[2] * P[2]
        
        D = torch.sqrt(xx+yy+zz)
                
        D = D.unsqueeze(0) ###to batched shape ##xyz: 1x3xhxw
                    
        return D

                
    def from_depth(self, d, xyz_sph):  
        ##input depth: 1xhxw  
                
        return  xyz_sph * d  ## (1x3xhxw) * (1xhxw)

    #####TEST
    def from_batched_depth(self, d, xyz_sph):  
        ##input depth: 1xhxw

        c_device = d.device

        db = d.unsqueeze(1)  
        return  db * xyz_sph.to(c_device)  ## (1x1xhxw) * (1x3xhxw) 

    #def euclidean_to_planar_depth(self, d, xz_sph):  
    #    ##input depth: hxw  
    #    c_device = d.device

    #    DP = d * xz_sph.to(c_device)  ## (hxw) (1xhxw) 
                                                
    #    return  DP ## (1xhxw)

    def euclidean_to_planar_depth(self, d, xz_sph, h_shift = None, v_shift = None):  
        ##input depth: 1xhxw  

        ##print(xz_sph.shape,d.shape)

        DP = xz_sph.to(d.device) * d  ## (1x3xhxw) * (1xhxw)

        if(h_shift is not None):               
            DP = torch.roll(DP, h_shift, dims=1)

        if(v_shift is not None):               
            DP = torch.roll(DP, v_shift, dims=0)

        return  DP 
                 
                    
    def to_depth(self, xyz, h_shift = None, v_shift = None):  
        D = np.zeros(shape =(xyz.shape[2],xyz.shape[3]),dtype=float) ##depth: 1xhxw

        x, y, z = torch.unbind(xyz, dim=1)

        xx = x * x
        yy = y * y
        zz = z * z

        D = torch.sqrt(xx+yy+zz)

        if(h_shift is not None):               
            D = torch.roll(D, h_shift, dims=1)

        if(v_shift is not None):               
            D = torch.roll(D, v_shift, dims=0)
                             
        if self.gpu:
            D = Variable(torch.FloatTensor(D)).cuda()
        else:
            D = Variable(torch.FloatTensor(D))
        
        D = D.unsqueeze(0) ### to batch shape 1x1xhxw
            
        return D 

    def to_planar_depth(self, xyz, h_shift = None, v_shift = None):  
        D = np.zeros(shape =(xyz.shape[2],xyz.shape[3]),dtype=float) ##depth: 1xhxw

        x, y, z = torch.unbind(xyz, dim=1)

        xx = x * x
        ##yy = y * y TO XZ plane - NB y is z in world coords
        zz = z * z

        D = torch.sqrt(xx+zz) 
        
        if(h_shift is not None):               
            D = torch.roll(D, h_shift, dims=1)

        if(v_shift is not None):               
            D = torch.roll(D, v_shift, dims=0)

        ##D = D - D1

        if self.gpu:
            D = Variable(torch.FloatTensor(D)).cuda()
        else:
            D = Variable(torch.FloatTensor(D))
        
        D = D.unsqueeze(0) ### to batch shape 1x1xhxw
                    
        return D 
          
    def get_surface_normal(self, xyz, patch_size=12):

        xyz = xyz.permute(0, 2, 3, 1) # [b, h, w, c]
                              
        # xyz: [1, h, w, 3]
        x, y, z = torch.unbind(xyz, dim=3)
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        z = torch.unsqueeze(z, 0)

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z

        if(self.gpu):
            patch_weight = torch.ones((1, 1, patch_size, patch_size), requires_grad=False).cuda()
        else:
            patch_weight = torch.ones((1, 1, patch_size, patch_size), requires_grad=False)
        
        xx_patch = nn.functional.conv2d(xx, weight=patch_weight, padding=int(patch_size / 2))
        yy_patch = nn.functional.conv2d(yy, weight=patch_weight, padding=int(patch_size / 2))
        zz_patch = nn.functional.conv2d(zz, weight=patch_weight, padding=int(patch_size / 2))
        xy_patch = nn.functional.conv2d(xy, weight=patch_weight, padding=int(patch_size / 2))
        xz_patch = nn.functional.conv2d(xz, weight=patch_weight, padding=int(patch_size / 2))
        yz_patch = nn.functional.conv2d(yz, weight=patch_weight, padding=int(patch_size / 2))

        ATA = torch.stack([xx_patch, xy_patch, xz_patch, xy_patch, yy_patch, yz_patch, xz_patch, yz_patch, zz_patch],
                          dim=4)
        ATA = torch.squeeze(ATA)
        ATA = torch.reshape(ATA, (ATA.size(0), ATA.size(1), 3, 3))
        eps_identity = 1e-6 * torch.eye(3, device=ATA.device, dtype=ATA.dtype)[None, None, :, :].repeat([ATA.size(0), ATA.size(1), 1, 1])
        ATA = ATA + eps_identity
        x_patch = nn.functional.conv2d(x, weight=patch_weight, padding=int(patch_size / 2))
        y_patch = nn.functional.conv2d(y, weight=patch_weight, padding=int(patch_size / 2))
        z_patch = nn.functional.conv2d(z, weight=patch_weight, padding=int(patch_size / 2))
        AT1 = torch.stack([x_patch, y_patch, z_patch], dim=4)
        AT1 = torch.squeeze(AT1)
        AT1 = torch.unsqueeze(AT1, 3)

        patch_num = 4
        patch_x = int(AT1.size(1) / patch_num)
        patch_y = int(AT1.size(0) / patch_num)

        n_img = torch.randn(AT1.shape).cuda()
        
        overlap = patch_size // 2 + 1

        for x in range(int(patch_num)):
            for y in range(int(patch_num)):
                left_flg = 0 if x == 0 else 1
                right_flg = 0 if x == patch_num -1 else 1
                top_flg = 0 if y == 0 else 1
                btm_flg = 0 if y == patch_num - 1 else 1
                at1 = AT1[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                      x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
                ata = ATA[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                      x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
                #try:
                n_img_tmp, _ = torch.solve(at1, ata)
                #except:
                #    print(at1, ata)
                n_img_tmp_select = n_img_tmp[top_flg * overlap:patch_y + top_flg * overlap, left_flg * overlap:patch_x + left_flg * overlap, :, :]
                n_img[y * patch_y:y * patch_y + patch_y, x * patch_x:x * patch_x + patch_x, :, :] = n_img_tmp_select

        ##n_img, _ = torch.solve(AT1, ATA)
        n_img_L2 = torch.sqrt(torch.sum(n_img ** 2, dim=2, keepdim=True))
        n_img_norm = n_img / n_img_L2

        # re-orient normals consistently
        ##FIXMEorient_mask = torch.sum(torch.squeeze(n_img_norm.cpu()) * torch.squeeze(xyz.cpu()), dim=2) > 0
        ##FIXMEn_img_norm[orient_mask] *= -1

        ###[h, w, c, b]
        ##restore batch first
        n_img_norm = n_img_norm.permute(3,2,0,1)
        
        ##n_img_norm = F.normalize(n_img_norm, dim=1)
        
        return n_img_norm

    def smoothness_from_depth(self, d, xz_sph):

        Dc  = self.euclidean_to_planar_depth(d, xz_sph)
        Dlu = self.euclidean_to_planar_depth(d, xz_sph, h_shift = -1, v_shift = -1)
        Drb = self.euclidean_to_planar_depth(d, xz_sph, h_shift = 1, v_shift = 1)
               
                        
        ###S = Dlu - 2 * Dc + Drb 2th order derivative
        
        S = - 2 * Dc + Dlu  + Drb
                            
        ##Smax = torch.max(S) 
        ##Smin = torch.min(S)

        ##range = Smax-Smin

        ##S = (S - Smin) / Smax
                
        ##S = F.normalize(S, dim=0)
        ##print('S min max',Smin, Smax)
        
        return S

    def batched_smoothness_from_depth(self, batch):
        batch_size = batch.size()[0]
        
        SM = []    
                        
        xz_sph = self.polar_sphere(batch.size()[1], batch.size()[2])
                        
        for i in range(batch_size):
            ##print('batch shape', batch[i:i+1].shape)
            sm = self.smoothness_from_depth(batch[i:i+1], xz_sph)
            SM.append(sm)
                                
        return torch.cat(SM, dim=0)

    def forward_batched_normals(self, batch, prj_type = 'euclidean'):#####input: Bxhxw
        batch_size = batch.size()[0]
                
        EPC_normals = []    
        
        if(prj_type == 'planar'):
            xyz_sph = self.polar_sphere(batch.size()[1], batch.size()[2])
        else:
            xyz_sph = self.xyz_sphere(batch.size()[1], batch.size()[2])
        
        for i in range(batch_size):
            ##print('batch shape', batch[i:i+1].shape)
            epc = self.from_depth(batch[i:i+1], xyz_sph)
            ##print('epc shape', epc.shape)
            nor = self.get_surface_normal(epc)
            EPC_normals.append(nor)
                                
        return torch.cat(EPC_normals, dim=0)

    def batched_euclidean_to_planar_depth(self, batch, canonical_tensor = False, atlanta_sphere = False):
        batch_size = batch.size()[0]
        
        DP = []

        H = batch.size()[1]
        W = batch.size()[2]
        
        if(canonical_tensor):
            H = batch.size()[2]
            W = batch.size()[3]

            batch = batch.squeeze(1)
        
        if(atlanta_sphere):
            xz_sph = self.atlanta_sphere(H, W)
        else:
            xz_sph = self.polar_sphere(H, W)
        
        for i in range(batch_size):
            ##print('batch shape', batch[i:i+1].shape)
            dp = self.euclidean_to_planar_depth(batch[i:i+1], xz_sph)
            DP.append(dp)

        result = torch.cat(DP, dim=0)

        if(canonical_tensor):
            result = result.unsqueeze(1)
                                
        return result
        

    def forward(self, batch):
        batch_size = batch.size()[0]
                
        EPC_batch = []   
        
        xyz_sph = self.xyz_sphere(batch.size()[1], batch.size()[2])
        
        for i in range(batch_size):
            epc = self.from_depth(batch[i:i+1], xyz_sph)
            EPC_batch.append(epc)
        
        self.EPC = torch.cat(EPC_batch, dim=0) ###to batch 

        ####TEST self.EPC = self.from_batched_depth(batch, xyz_sph)
                                
        return self.EPC

if __name__ == '__main__':
    print('testing EPC')
    device = torch.device('cpu')  
    
    depth_batch = torch.ones([4, 256, 512]).to(device)
    
    epc = EPC(gpu=False)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    pc_batch = epc(depth_batch)
    ##pc_batch = epc.forward_batched_normals(depth_batch)
    
    ##pc_out = epc.batched_smoothness_from_depth(depth_batch)##epc.to_planar_depth(pc_batch)
    end.record()

    
    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('time cost',start.elapsed_time(end))

    print('output shape',pc_batch.shape)

