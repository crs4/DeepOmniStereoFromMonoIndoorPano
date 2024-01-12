import os

import torch
import torch.nn as nn

from gated_sean_model import GatedNet, get_z_encoding
from misc import tools, eval

###NEW
from slicenet_gated_model_scalable import GatedSliceNet, SegNet
##from slicenet_model_scalable import SliceNet
from geometry.render import *
from geometry.panorama import *
from gated_modules import *

from misc.layout import D2L

###DEBUG
from thop import profile, clever_format

       
class IndoorSynthNet(nn.Module):
    def __init__(self, device='none', backbone='light_rgbe', decoder_type='rgbd', full_size = False, bypass = False, compose_out = False, masked_layout = True, slice_w = 32):
        super(IndoorSynthNet, self).__init__()

        self.decoder_type = decoder_type
        self.backbone = backbone
        self.full_size = full_size ##used to set layout estimation scalability  
        
        self.bypass = bypass
        self.compose_out = compose_out
        self.masked_layout = masked_layout

        self.slice_w = slice_w

        self.get_mapping = False
        
        print('IndoorSynthNet',self.backbone,self.decoder_type)
                     
                
        if(self.backbone == 'light_rgbe' or self.backbone == 'light_rgbe_seg'):
            if(self.decoder_type == 'rgb_comp'):
                self.image_synth = GatedNet(device, backbone=self.backbone, decoder_type='rgb') ##
            else:
                self.image_synth = GatedNet(device, backbone=self.backbone, decoder_type=self.decoder_type) ##

        if(self.backbone == 'light_rgbe_dr'):
            if(self.decoder_type == 'rgb_comp'):
                self.image_synth = GatedNet(device, backbone='light_rgbe', decoder_type='rgb') ##
            else:
                self.image_synth = GatedNet(device, backbone='light_rgbe', decoder_type=self.decoder_type) ##

        if(self.backbone == 'rgbe_dr_sliced_sink'):
            self.slice_w = 160### 128 + 32
            self.s_factor = 2
            self.image_synth = GatedNet(device, backbone=self.backbone, decoder_type=self.decoder_type) ##  

        if(self.backbone == 'rgbe_dr_sliced'):
            self.slice_w = 160##FIXME32### 128 + 32
            self.image_synth = GatedNet(device, backbone=self.backbone, decoder_type=self.decoder_type) ##  
            
        if(self.backbone == 'light_rgbe_dr_sliced'):
            self.slice_w = 160###
            self.image_synth = GatedNet(device, backbone=self.backbone, decoder_type=self.decoder_type) ##
            
        if(self.backbone == 'light_rgb' and self.decoder_type == 'depth_layout'):

            print('IndoorSynth backbone',self.backbone, 'decoder', self.decoder_type)

            self.image_synth = GatedNet(device, self.backbone, decoder_type='depth') ##
            
            if(not self.full_size):
                print('using fixed size layout trasform depth input', 512, 1024)
                self.d2l = D2L(gpu=True, H = 512, W = 1024)####fixed size transforms

            self.layout_from_depth = SegNet(backbone='light_depth')
        
        self.tr_img= None
       
       
    def forward(self, img, tr = None, img_depth = None, edges = None, src_sem_layout = None, trg_sem_layout = None, style_codes = None, slice_c = 512):
        ###
        b, c, h, w = img.size()
                
        if(self.backbone == 'light_rgbe'):                    
            #common case  
            tr_img, tr_mask = render(img, img_depth.unsqueeze(1), tr, max_depth=20, get_mask = True, masked_img = True, filter_iter = 1, masks_th = 0.9)
            ##tr_img, tr_mask = render(img, img_depth.unsqueeze(1), tr, max_depth=20, get_mask = True, masked_img = True, filter_iter = 0, masks_th = 0.0)
            tr_mask_inverse = 1 - tr_mask #####inverse: parts to be filled

            if(self.bypass):
                return tr_img, tr_mask
                        
              
            if(edges == None):
                edges = torch.zeros_like(tr_mask_inverse).to(tr_mask_inverse.device)

            if(self.masked_layout):
                edges = tr_mask_inverse*edges

            proc_img = torch.cat((tr_img, edges), 1) #### N channels input

            output = self.image_synth(tr_img, tr_mask_inverse, proc_img) ####first image not used here - third element to be processed   
            
            if (self.compose_out): #####LEGACY DEBUG output with rgbd output - force to compose pre-trained image synth
                temp_out = output[:,:3]
                output[:,:3] = (0.7*tr_img+0.3*(tr_mask*temp_out)) + tr_mask_inverse*temp_out

            if (self.decoder_type == 'rgb_comp'): ###trainable composition of
                temp_out = output
                output = tr_img*tr_mask + temp_out*tr_mask_inverse


            return output, tr_mask

        if(self.backbone == 'light_rgbe_dr'):                    
            #common case

            ##print('input',img.shape, img_depth.shape)
            if(self.get_mapping):
                tr_img, tr_mask, mapping = simple_gpu_renderer(img, img_depth.unsqueeze(1), tr, get_mapping = self.get_mapping)
            else:
                tr_img, tr_mask = simple_gpu_renderer(img, img_depth.unsqueeze(1), tr, get_mapping = self.get_mapping)
           
            ##print('output',tr_img.shape, tr_mask.shape)
            tr_mask_inverse = 1 - tr_mask #####inverse: parts to be filled

            if(self.bypass):
                return tr_img, tr_mask                        
              
            if(edges == None):
                edges = torch.zeros_like(tr_mask_inverse).to(tr_mask_inverse.device)

            if(self.masked_layout):
                edges = tr_mask_inverse*edges

            proc_img = torch.cat((tr_img, edges), 1) #### N channels input

            output = self.image_synth(tr_img, tr_mask_inverse, proc_img) ####first image not used here - third element to be processed   
            
            if (self.compose_out): #####LEGACY DEBUG output with rgbd output - force to compose pre-trained image synth
                temp_out = output[:,:3]
                output[:,:3] = (0.7*tr_img+0.3*(tr_mask*temp_out)) + tr_mask_inverse*temp_out

            if (self.decoder_type == 'rgb_comp'): ###trainable composition of
                temp_out = output
                output = tr_img*tr_mask + temp_out*tr_mask_inverse

            if(self.get_mapping):
                return output, tr_mask, mapping
            else:
                return output, tr_mask         
        
        if(self.backbone == 'light_rgb' and self.decoder_type=='depth_layout'):
            ###attended output: Bxhxw depth, Bxfwxfw mask
            empty_invalid_mask = torch.zeros(b, 1, h, w).to(img.device)### back-compatibility with image synth format
            d = self.image_synth(img, empty_invalid_mask, img) #### NB. proc_img==img has 3 channels - back-compatibility with image synth format

            if(self.full_size):
                self.d2l = D2L(gpu=True, H = h, W = w)####NB slower but scalable

            x_atl_depth, d_down, c_dist, f_dist, fl = self.d2l.batched_atlanta_transform_from_depth(d)#####input Bx1xhxw
            
            ##print('output atl depth', x_atl_depth.shape, x_atl_depth.device)
            layout = self.layout_from_depth(x_atl_depth) ### input: Bx1xhxh

            result = []

            result.append(d.squeeze(1))  
            result.append(layout)

            return result   
        
        if(self.backbone == 'light_rgbe_dr_sliced' or self.backbone == 'rgbe_dr_sliced'):                    
            #NB. rendering a larger slice: slice_w * 4

            ##print('input',img.shape, img_depth.shape)
            if(self.get_mapping):
                tr_img, tr_mask, mapping = simple_gpu_renderer(img, img_depth.unsqueeze(1), tr, get_mapping = self.get_mapping, slice_w = (self.slice_w), slice_col = slice_c)
            else:
                tr_img, tr_mask = simple_gpu_renderer(img, img_depth.unsqueeze(1), tr, get_mapping = self.get_mapping, slice_w = (self.slice_w), slice_col = slice_c)
           
            ##print('output',tr_img.shape, tr_mask.shape)
            tr_mask_inverse = 1 - tr_mask #####inverse: parts to be filled

            if(self.bypass):
                return tr_img, tr_mask                        
              
            if(edges == None):
                edges = torch.zeros_like(tr_mask_inverse).to(tr_mask_inverse.device)

            if(self.masked_layout):
                edges = tr_mask_inverse*edges

            proc_img = torch.cat((tr_img, edges), 1) #### N channels input

            output = self.image_synth(tr_img, tr_mask_inverse, proc_img) ####first image not used here - third element to be processed  
            #
            ##output = self.image_synth(tr_img, tr_mask_inverse, tr_img) 
            
            if (self.compose_out): #####LEGACY DEBUG output with rgbd output - force to compose pre-trained image synth
                temp_out = output[:,:3]
                output[:,:3] = (0.7*tr_img+0.3*(tr_mask*temp_out)) + tr_mask_inverse*temp_out

            if (self.decoder_type == 'rgb_comp'): ###trainable composition of
                temp_out = output
                output = tr_img*tr_mask + temp_out*tr_mask_inverse

            if(self.get_mapping):
                return output, tr_mask, mapping
            else:
                return output, tr_mask 

        if(self.backbone == 'rgbe_dr_sliced_sink'):                    
            #NB. rendering a larger slice: slice_w * 4

            ##print('input',img.shape, img_depth.shape)
            if(self.get_mapping):
                tr_img, tr_mask, mapping = simple_gpu_renderer(img, img_depth.unsqueeze(1), tr, get_mapping = self.get_mapping, slice_w = (self.slice_w*self.s_factor), slice_col = slice_c)
            else:
                tr_img, tr_mask = simple_gpu_renderer(img, img_depth.unsqueeze(1), tr, get_mapping = self.get_mapping, slice_w = (self.slice_w*self.s_factor), slice_col = slice_c)
           
            ##print('output',tr_img.shape, tr_mask.shape)
            tr_mask_inverse = 1 - tr_mask #####inverse: parts to be filled

            if(self.bypass):
                return tr_img, tr_mask                      
                       
            if(edges == None):
                edges = torch.zeros_like(tr_mask_inverse).to(tr_mask_inverse.device)

            if(self.masked_layout):
                edges = tr_mask_inverse*edges

            proc_img = torch.cat((tr_img, edges), 1) #### N channels input

            output = self.image_synth(tr_img, tr_mask_inverse, proc_img)  ####first image not used here - third element to be processed   
            
            if (self.compose_out): #####LEGACY DEBUG output with rgbd output - force to compose pre-trained image synth
                temp_out = output[:,:3]
                output[:,:3] = (0.7*tr_img+0.3*(tr_mask*temp_out)) + tr_mask_inverse*temp_out

            if (self.decoder_type == 'rgb_comp'): ###trainable composition of
                temp_out = output
                output = tr_img*tr_mask + temp_out*tr_mask_inverse

            if(self.get_mapping):
                return output, tr_mask, mapping
            else:
                return output, tr_mask 
       
  

def counter():
    print('testing IndoorSynth')

    ##os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    device = torch.device('cuda')

    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    test_full = True
       
    ##net = IndoorSynthNet(device, decoder_type='rgbd', backbone ='light_rgbe', bypass = False).to(device)
    ##net = IndoorSynthNet(device, decoder_type='rgb', backbone ='light_rgbe', bypass = False).to(device)
    ##net = IndoorSynthNet(device, decoder_type='rgb_comp', backbone ='light_rgbe', bypass = False).to(device)
    net = IndoorSynthNet(device, decoder_type='rgb', backbone ='rgbe_dr_sliced_sink', bypass = False).to(device)
    ##net = IndoorSynthNet(device, decoder_type='rgb', backbone ='rgb_dr_sliced', bypass = False).to(device)    
    
    # testing
    layers = net

    inputs = []

    B = 1

    ##H,W = 1024, 2048
    ##H,W = 2048,4096
    H,W = 512,1024

    if(test_full):        
        img = torch.randn(B, 3, H, W).to(device)
        inputs.append(img)
        #
    else:
        H,W = 256, 512
        img = torch.randn(B, 3, 256, 512).to(device)
        inputs.append(img)

    tr = torch.randn(B, 1, 3).to(device)
    inputs.append(tr)

    gt_depth = torch.randn(B, H, W).to(device) ####FIXME
           
    inputs.append(gt_depth)
                
    ##out = layers(img,mask,masked_input,device)

    with torch.no_grad():
        flops, params = profile(layers, inputs)
    ##print(f'input :', [v.shape for v in inputs])
    print(f'flops : {flops/(10**9):.2f} G')
    print(f'params: {params/(10**6):.2f} M')

    import time
    fps = []
    with torch.no_grad():
        out,_ = layers(img,tr,gt_depth)
        print('out shape',out.shape)

        for _ in range(50):
            eps_time = time.time()
            layers(img,tr,gt_depth)
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
    print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')

def combo_counter():
    print('testing gated combonet')

    ##os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    device = torch.device('cuda')

    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    test_full = True
        
    net = IndoorSynthNet(device, decoder_type='depth_layout', backbone ='light_rgb', full_size = True).to(device)
        
    # testing
    layers = net

    inputs = []

    B = 1

    H,W = 512, 1024

    if(test_full):
        
        img = torch.randn(B, 3, H,W).to(device)
        inputs.append(img)
        #
    else:
        
        img = torch.randn(B, 3, 256, 512).to(device)
        inputs.append(img)

    ##tr = torch.randn(B, 1, 3).to(device)
    ##inputs.append(tr)

    ##gt_depth = torch.randn(B, H,W).to(device) ####FIXME


    ##gt_edges = torch.randn(B, 1, H,W).to(device) ####FIXME

    ##inputs.append(gt_depth)

    ##inputs.append(gt_edges)
        
    ##out = layers(img,mask,masked_input,device)

    #with torch.no_grad():
    #    flops, params = profile(layers, inputs)
    ##print(f'input :', [v.shape for v in inputs])
    #print(f'flops : {flops/(10**9):.2f} G')
    #print(f'params: {params/(10**6):.2f} M')

    import time
    fps = []
    with torch.no_grad():
        d,l = layers(img)
        print('out shape',d.shape, l.shape)

        for _ in range(50):
            eps_time = time.time()
            layers(img)
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
    print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')


if __name__ == '__main__':
    ##device = torch.device('cuda')
    ##net = EmptyingRoomNet(device, decoder_type='rgbd', backbone ='light').to(device)

    ##combo_counter()

    counter()
        
    ##sean_counter()

    ##splat_counter()

