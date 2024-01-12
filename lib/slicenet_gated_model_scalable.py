import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools
import time

import os
import math

from mhsa_pos import MHSATransformerPos

from resnet_mod import resnet18_single_channel, resnet18_single_channel_gated,resnet18_rgbs_channel_gated, resnet34_rgbs_channel_gated, resnet18_channels

from gated_modules import *

from thop import profile, clever_format

from geometry.render import *

#####new layout integration
from gated_sean_model import SegNet
from misc.layout import D2L

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]

''' Pad left/right-most to each other instead of zero padding '''
def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)

class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)

def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )
        #############################################################

def get_segmentation_masks(seg_pred):
    soft_sem = torch.softmax(seg_pred, dim = 1) #####TO DO - here semantic is given by clutter mask
    soft_sem = torch.argmax(soft_sem, dim=1, keepdim=True)
    soft_sem = torch.clamp(soft_sem, min=0, max=1)
    masks = torch.zeros_like(seg_pred).to(seg_pred.device)
    masks.scatter_(1, soft_sem, 1)

    #filter_iter = 1

    #for i in range(filter_iter): 
    #    b,c,h,w = masks.size()
    #    masks = F.interpolate(masks, size=(h//2, w//2), mode='bilinear', align_corners=False)
    #    masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)
        
    return masks
		
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool
                
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4     

class AConv(nn.Module):
    def __init__(self, in_c, out_c, ks=3, st=(2, 1), gated_conv = False):
        super(AConv, self).__init__()
        assert ks % 2 == 1
        if(gated_conv):
            self.layers = GatedConv2d(in_c, out_c, kernel_size=ks, stride=st, padding=ks//2, pad_type = 'spherical', activation = 'elu')
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, stride=st, padding=ks//2),
                ##nn.BatchNorm2d(out_c),
                ##nn.PReLU(out_c),
                nn.ELU(inplace=True),
            )

    def forward(self, x):
        return self.layers(x)
    
class Slicing(nn.Module):
    def __init__(self, in_c, out_c, st=(2, 1), gated_conv = False):
        super(Slicing, self).__init__()

        ##print(out_c)

        self.layer = nn.Sequential(
            AConv(in_c, in_c//2, st=st, gated_conv=gated_conv),
            AConv(in_c//2, in_c//4, st=st, gated_conv=gated_conv),
            AConv(in_c//4, out_c, st=st, gated_conv=gated_conv),
        )
        
    def forward(self, x, out_w, x_depth=None, tr = None):
        if(tr != None and x_depth != None):
            print('translate',tr.shape,x_depth.shape)
            x = render(x, x_depth, tr, max_depth=20, masked_img = True) 

        x = self.layer(x)
        
        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]

        if(x.shape[2] != out_w):  
            ##print('using interpolation to', out_w)
            #####HorizonNet-style upsampling        
            x = torch.cat([x[..., -1:], x, x[..., :1]], 3) ## plus 2 on W
            x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False) ####NB interpolating only W
            x = x[..., factor:-factor] ##minus 2 on W

            ##SIMPLEST
            ##x = F.interpolate(x, size=(x.shape[2], out_w), mode='bilinear', align_corners=False)
                                                                   
        return x

class MultiSlicing(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8, stride=(2,1), gated_conv = False):
        super(MultiSlicing, self).__init__()
        self.cs = c1, c2, c3, c4 
        
        self.out_scale = out_scale
        self.slc_lst = nn.ModuleList([
            Slicing(c1, c1//out_scale, st=stride, gated_conv = gated_conv), 
            Slicing(c2, c2//out_scale, st=stride, gated_conv = gated_conv), 
            Slicing(c3, c3//out_scale, st=stride, gated_conv = gated_conv),
            Slicing(c4, c4//out_scale, st=stride, gated_conv = gated_conv),
        ])

        
    def forward(self, conv_list, out_w, no_cat = False): ###out_w drives interpolation
        ##print(len(conv_list))
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]

        if(no_cat):
            feature = []

            for f, x, out_c in zip(self.slc_lst, conv_list, self.cs):
                _,_,_,w = x.size()
                feature.append( f(x,w).reshape(bs, -1, w) )
                              
        else:            
            feature = torch.cat([
                f(x, out_w).reshape(bs, -1, out_w)
                for f, x, out_c in zip(self.slc_lst, conv_list, self.cs)
            ], dim=1)        
            
        return feature

class MultiSlicingTrans(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8, stride=(2,1), gated_conv = True):
        super(MultiSlicingTrans, self).__init__()
        self.cs = c1, c2, c3, c4 
        
        self.out_scale = out_scale
        self.slc_lst = nn.ModuleList([
            Slicing(c1, c1//out_scale, st=stride, gated_conv = gated_conv), 
            Slicing(c2, c2//out_scale, st=stride, gated_conv = gated_conv), 
            Slicing(c3, c3//out_scale, st=stride, gated_conv = gated_conv),
            Slicing(c4, c4//out_scale, st=stride, gated_conv = gated_conv),
        ])

        
    def forward(self, conv_list, out_w, x_depth, tr, no_cat = False): ###out_w drives interpolation
        ##print(len(conv_list))
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]

        if(no_cat):
            feature = []

            for f, x, out_c in zip(self.slc_lst, conv_list, self.cs):
                _,_,_,w = x.size()
                feature.append( f(x,w,x_depth,tr).reshape(bs, -1, w) )
                              
        else:            
            feature = torch.cat([
                f(x, out_w, x_depth,tr).reshape(bs, -1, out_w)
                for f, x, out_c in zip(self.slc_lst, conv_list, self.cs)
            ], dim=1)        
            
        return feature

class GatedSliceNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, full_size = True, sparse_encoder = False, compression_type='mhsa', ref_type = 'none', max_size = 1024):
        super(GatedSliceNet, self).__init__()
        self.backbone = backbone
        self.ch_scale = 4  
        
        self.ref_type = ref_type

        print('using ref', self.ref_type)
                
        self.gated_channels = 5

        if(self.ref_type == 'mask_one' or self.ref_type == 'rgbs2rgbd'):
            self.gated_channels = 4

        if(self.ref_type == 'rgbd'):
            self.gated_channels = 3
        
        
        h,w = max_size//2,max_size
        self.max_enc = max_size 

        self.full_size = full_size ####NOT USED   
        
        
        self.compression_type = compression_type ####NB. sed for sparse feature refinement

        print('using compression', self.compression_type)
        
        self.sparse_encoder = sparse_encoder

        #####features eencoder block
        
        print('using single branch rgbs encoder')

        if(self.backbone == 'resnet18'):
            self.feature_extractor_spr = resnet18_rgbs_channel_gated(inplanes = 64, input_channels=self.gated_channels) #### - set 4 channels + mask

        if(self.backbone == 'resnet34'):
            self.feature_extractor_spr = resnet34_rgbs_channel_gated(inplanes = 64, input_channels=self.gated_channels) #### 
                
        # Inference channels number from each block of the encoder
        with torch.no_grad():
            #dummy = torch.zeros(1, self.gated_channels, h, w)##NB c1, c2, c3, c4 do not depend by resolution
            #c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor_spr(dummy)] ###NB depend by resnet layers depth                                    
            #self.lfeats = (c1*8 + c2*4 + c3*2 + c4*1) // self.ch_scale            
            ####TO DO replace dummy etc.
            ##self.lfeats *= (w//512)#####FIXME - hires support

            c1, c2, c3, c4 = 64, 128, 256, 512
            self.lfeats = w

            ##print('latent feat',self.lfeats)
        ##########################################
        
        #####features compression block###########################
                
        self.stride=(2,1)
        self.slicing_module_spr = MultiSlicing(c1, c2, c3, c4, self.ch_scale, stride=self.stride)###NB. not using gating in slicing, gated_conv = self.sparse_gated_encoder)
                             
        ##################################################################
         
        #####feature refinement block##################################################
        
        if(self.compression_type == 'mhsa'):
            print('using mhsa for psarse branch')
            self.mhsa_spr = MHSATransformerPos(num_layers=1, d_model=self.lfeats, num_heads=4, conv_hidden_dim=self.lfeats//2, maximum_position_encoding = self.max_enc)  

        if(self.compression_type == 'mhsa_no_inter'):
            print('using mhsa no inter for psarse branch')
            self.mhsa_spr = MHSATransformerPos(num_layers=1, d_model=self.lfeats, num_heads=4, conv_hidden_dim=self.lfeats//2, maximum_position_encoding = self.max_enc//4)
        ################################################################################

        self.drop_out = nn.Dropout(0.5)        

        self.decoder1 = nn.ModuleList([])
                                
        #if(full_size):
        dec_count = int(math.log2(w))-1

        print('dec count',dec_count)

        last_ch = 1
         
        if(self.ref_type == 'rgbd' or self.ref_type == 'rgbs2rgbd'):
            ### decoder 1 common for rgb and d since dec_count-1
            for i in range(dec_count-1):
                #                
                sf1 = pow(2,i)
                factor = 2
                                       
                                                                               
                sf2 = factor*sf1                                
                                
                ch_in  = self.lfeats // sf1
                ch_out = self.lfeats // sf2
                                
                #if((i == dec_count-2)):
                #    ch_in = last_ch
                #    ch_out = last_ch
                
                last_ch = ch_out
                                           

                ##print('scale ch', ch_in, ch_out)                                
                                
                decoder1 = nn.Sequential(
                            AConv(ch_in, ch_out, st=(1, 1)), ##gated_conv = self.sparse_gated_encoder),                            
                        )                
                                
                self.decoder1.append(decoder1)

            ##print('ch_out',ch_out)

            self.decoder_d   = GatedConv2d(ch_out, 1, kernel_size=3, stride=(1,1), padding=3//2, pad_type = 'spherical', activation = 'elu')
            self.decoder_rgb = GatedConv2d(ch_out, 3, kernel_size=3, stride=(1,1), padding=3//2, pad_type = 'spherical', activation = 'tanh')
        else:
            print('only depth mode')
            for i in range(dec_count):
                #                
                sf1 = pow(2,i)
                                                               
                sf2 = 2*sf1
                                
                ch_in  = self.lfeats // sf1
                ch_out = self.lfeats // sf2

                if(i == dec_count-1):
                    ch_out = 1                    

                ##print('scale ch', ch_in, ch_out)
                                
                                
                decoder1 = nn.Sequential(
                            AConv(ch_in, ch_out, st=(1, 1)), ##gated_conv = self.sparse_gated_encoder),                            
                        )
                
                                
                self.decoder1.append(decoder1)
            
        if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):
            self.s_lst = []
            for i in range(dec_count+1):
                ups = (2,1)
                if(i<2):
                    ups = (2,2)
                self.s_lst.append(ups)
               

                              
        ''' Pad left/right-most to each other instead of zero padding '''       
        ####FIXME - use only for rgb modules - gated sparse already have
        wrap_lr_pad(self)             
    
    
    def _prepare_rgb(self, x):
        ####CHECK IT for sparse input
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
                    
        return (x[:, :3] - self.x_mean) / self.x_std

    def get_latent_feature(self, x_rgb, x_spr, no_cat=False, x_rgb_masked = None):
        x_rgb = self._prepare_rgb(x_rgb)               
       
        if(self.ref_type == 'rgbd'):
            x_spr_m = self._prepare_rgb(x_rgb_masked) 
            ###x_spr_m = torch.cat([x_rgb_masked, x_spr],dim=1) ### 4 channels
        else:
            x_in = torch.cat([x_rgb, x_spr],dim=1) ### 4 channels
            ###default mask pixels
            x_valid = (x_spr>0).float()
            #### build masked img and invalid mask
            x_invalid = 1 - x_valid                

            if(self.ref_type == 'no_masking' or self.ref_type == 'mask_one'):
                x_masked = x_in
            else:
                x_masked = x_valid*x_in + x_invalid ### 4 channels
                             
            ##print('print test in shape x_masked',x_masked.shape)

            if(self.ref_type == 'mask_one'):
                x_spr_m = x_in
            else:
                x_spr_m = torch.cat([x_masked, x_invalid],dim=1) ##5 channels ######

            ####FIXME - override
            if(self.ref_type == 'mask_sparse'):
                x_spr_m = torch.cat([x_in, x_valid],dim=1) ##5 channels


        ##print('gated encoder in',x_spr_m.shape)

        conv_list_spr = self.feature_extractor_spr(x_spr_m)  ####5 channels - for sparse2dense - 4 channels for image synth
        
                
        if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):
            feature_spr = self.slicing_module_spr(conv_list_spr, conv_list_spr[0].shape[3], no_cat = no_cat) ##### out_w = x_spr.shape[3] : force interpolation to max_w

            ##print('feature_sparse',feature_spr.shape)
        if(self.compression_type == 'none'): ####FIXME - for back compatibilty
            feature_spr = self.slicing_module_spr(conv_list_spr, x_spr.shape[3], no_cat = nocat) ##### out_w = x_spr.shape[3] : force interpolation to max_w
                                   
        ##print(feature_spr[0].shape,feature_spr[1].shape,feature_spr[2].shape,feature_spr[3].shape)

        return feature_spr ##return BxDxW0 i.e. Bx1024x256

    
    def forward(self, x_rgb, x_spr, x_rgb_masked = None):
        x_rgb = self._prepare_rgb(x_rgb)               
       
        if(self.ref_type == 'rgbd'):
            x_spr_m = self._prepare_rgb(x_rgb) ####NB. only 3 channels - sparse depth is missing
            ###x_spr_m = torch.cat([x_rgb_masked, x_spr],dim=1) ### 4 channels
        else:
            x_in = torch.cat([x_rgb, x_spr],dim=1) ### 4 channels - also for case rgbs2rgbd NB. expecting x_spr is the translated depth
                        
            if(self.ref_type == 'no_masking' or self.ref_type == 'mask_one'):
                x_masked = x_in
            else:
                ###default mask pixels
                x_valid = (x_spr>0).float()
                #### build masked img and invalid mask
                x_invalid = 1 - x_valid  
                x_masked = x_valid*x_in + x_invalid ### 4 channels
                             
            ##print('print test in shape x_masked',x_masked.shape)

            if(self.ref_type == 'mask_one' or self.ref_type == 'rgbs2rgbd'):
                x_spr_m = x_in
            else:
                x_spr_m = torch.cat([x_masked, x_invalid],dim=1) ##5 channels ######

            ####FIXME - override
            if(self.ref_type == 'mask_sparse'):
                x_spr_m = torch.cat([x_in, x_valid],dim=1) ##5 channels


        ##print('gated encoder in',x_spr_m.shape)

        conv_list_spr = self.feature_extractor_spr(x_spr_m)  ####5 channels - for sparse2dense - 4 channels for image synth
        
                
        if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):
            feature_spr = self.slicing_module_spr(conv_list_spr, conv_list_spr[0].shape[3]) ##### out_w = x_spr.shape[3] : force interpolation to max_w

            ##print('feature_sparse',feature_spr.shape)
        if(self.compression_type == 'none'): ####FIXME - for back compatibilty
            feature_spr = self.slicing_module_spr(conv_list_spr, x_spr.shape[3]) ##### out_w = x_spr.shape[3] : force interpolation to max_w
            ##print('feature_sparse',feature_spr.shape)
                  
               
        if(self.compression_type == 'mhsa' or self.compression_type == 'mhsa_no_inter'):
            ##print('using mhsa')
            feature_spr = feature_spr.permute(0, 2, 1)  # [b, w, c*h]
            output_spr = self.mhsa_spr(feature_spr)                                 
            output_spr = self.drop_out(output_spr)
            output_spr = output_spr.permute(0, 2, 1) ###restore order
        else:
            ##print('using no mhsa')
            output_spr = feature_spr 
                
        output_spr = output_spr.reshape(output_spr.shape[0], output_spr.shape[1], 1, output_spr.shape[2])
                  

        for i in range(len(self.decoder1)): 
            if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):            
                output_spr = F.interpolate(output_spr, scale_factor=self.s_lst[i], mode='nearest')  
            else:
                output_spr = F.interpolate(output_spr, scale_factor=self.stride, mode='nearest')
            
            output_spr = self.decoder1[i](output_spr) 
            
        if(self.ref_type == 'rgbd' or self.ref_type == 'rgbs2rgbd'):
            output_spr = F.interpolate(output_spr, scale_factor=self.s_lst[i], mode='nearest') ####Bx4xhxw

            out_rgb = self.decoder_rgb(output_spr)
            out_d = self.decoder_d(output_spr)
            
            out_rgb = torch.clamp(out_rgb, 0, 1) ####CHECK IT

            ##print(out_rgb.shape,out_d.shape)
                            
            output_spr = torch.cat([out_rgb,out_d],dim=1)
            
                                                 
        output_spr = output_spr.squeeze(1)
                   
        
                                                                   
        return output_spr

class SliceNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, full_size = True, sparse_encoder = False, compression_type='mhsa', ref_type = 'none', max_size = 1024):
        super(SliceNet, self).__init__()
        self.backbone = backbone
        self.ch_scale = 4  
        
        self.ref_type = ref_type

        print('using ref', self.ref_type)
                
        self.gated_channels = 3

        if(self.ref_type == 'mask_one' or self.backbone == 'resnet18_rgbd'):
            self.gated_channels = 4

        if(self.ref_type == 'rgbd'):
            self.gated_channels = 3
        
        
        h,w = max_size//2,max_size
        self.max_enc = max_size 

        self.full_size = full_size ####NOT USED   
        
        
        self.compression_type = compression_type ####NB. sed for sparse feature refinement

        print('using compression', self.compression_type)
        
        self.sparse_encoder = sparse_encoder

        #####features eencoder block
        
        print('using single branch rgbs encoder')

        if(self.backbone == 'resnet18' or self.backbone == 'resnet18_rgbd'):
            self.feature_extractor_spr = resnet18_channels(input_channels=self.gated_channels) #### - set 4 channels + mask
                   
                
        # Inference channels number from each block of the encoder
        with torch.no_grad():
            #dummy = torch.zeros(1, self.gated_channels, h, w)##NB c1, c2, c3, c4 do not depend by resolution
            #c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor_spr(dummy)] ###NB depend by resnet layers depth                                    
            #self.lfeats = (c1*8 + c2*4 + c3*2 + c4*1) // self.ch_scale            
            ####TO DO replace dummy etc.
            ##self.lfeats *= (w//512)#####FIXME - hires support

            c1, c2, c3, c4 = 64, 128, 256, 512
            self.lfeats = w

            ##print('latent feat',self.lfeats)
        ##########################################
        
        #####features compression block###########################
                
        self.stride=(2,1)
        self.slicing_module_spr = MultiSlicing(c1, c2, c3, c4, self.ch_scale, stride=self.stride)###NB. not using gating in slicing, gated_conv = self.sparse_gated_encoder)
                             
        ##################################################################
         
        #####feature refinement block##################################################
        
        if(self.compression_type == 'mhsa'):
            print('using mhsa for psarse branch')
            self.mhsa_spr = MHSATransformerPos(num_layers=1, d_model=self.lfeats, num_heads=4, conv_hidden_dim=self.lfeats//2, maximum_position_encoding = self.max_enc)  

        if(self.compression_type == 'mhsa_no_inter'):
            print('using mhsa no inter for psarse branch')
            self.mhsa_spr = MHSATransformerPos(num_layers=1, d_model=self.lfeats, num_heads=4, conv_hidden_dim=self.lfeats//2, maximum_position_encoding = self.max_enc//4)
        ################################################################################

        self.drop_out = nn.Dropout(0.5)        

        self.decoder1 = nn.ModuleList([])
                                
        #if(full_size):
        dec_count = int(math.log2(w))-1

        print('dec count',dec_count)

        last_ch = 1
         
        if(self.ref_type == 'rgbd'):
            ### decoder 1 common for rgb and d since dec_count-1
            for i in range(dec_count-1):
                #                
                sf1 = pow(2,i)
                factor = 2
                                       
                                                                               
                sf2 = factor*sf1                                
                                
                ch_in  = self.lfeats // sf1
                ch_out = self.lfeats // sf2
                                
                #if((i == dec_count-2)):
                #    ch_in = last_ch
                #    ch_out = last_ch
                
                last_ch = ch_out
                                           

                ##print('scale ch', ch_in, ch_out)                                
                                
                decoder1 = nn.Sequential(
                            AConv(ch_in, ch_out, st=(1, 1)), ##gated_conv = self.sparse_gated_encoder),                            
                        )                
                                
                self.decoder1.append(decoder1)

            ##print('ch_out',ch_out)

            self.decoder_d   = GatedConv2d(ch_out, 1, kernel_size=3, stride=(1,1), padding=3//2, pad_type = 'spherical', activation = 'elu')
            self.decoder_rgb = GatedConv2d(ch_out, 3, kernel_size=3, stride=(1,1), padding=3//2, pad_type = 'spherical', activation = 'tanh')
        else:
            print('only depth mode')
            for i in range(dec_count):
                #                
                sf1 = pow(2,i)
                                                               
                sf2 = 2*sf1
                                
                ch_in  = self.lfeats // sf1
                ch_out = self.lfeats // sf2

                if(i == dec_count-1):
                    ch_out = 1                    

                ##print('scale ch', ch_in, ch_out)
                                
                                
                decoder1 = nn.Sequential(
                            AConv(ch_in, ch_out, st=(1, 1)), ##gated_conv = self.sparse_gated_encoder),                            
                        )
                
                                
                self.decoder1.append(decoder1)
            
        if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):
            self.s_lst = []
            for i in range(dec_count+1):
                ups = (2,1)
                if(i<2):
                    ups = (2,2)
                self.s_lst.append(ups)
               

                              
        ''' Pad left/right-most to each other instead of zero padding '''       
        ####FIXME - use only for rgb modules - gated sparse already have
        wrap_lr_pad(self)             
    
    
    def _prepare_rgb(self, x):
        ####CHECK IT for sparse input
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
                    
        return (x[:, :3] - self.x_mean) / self.x_std
        
    
    def forward(self, x_rgb, x_depth = None):
        x_rgb = self._prepare_rgb(x_rgb)               
       
        if(self.ref_type == 'rgbd'):
            x_spr_m = self._prepare_rgb(x_rgb) 
            ###x_spr_m = torch.cat([x_rgb_masked, x_spr],dim=1) ### 4 channels

        if(self.backbone == 'resnet18'):
            x_spr_m = x_rgb

        if(self.backbone == 'resnet18_rgbd'):
            x_spr_m = torch.cat([x_rgb, x_depth],dim=1) ### 4 channels
                                        
            ##print('print test in shape x_masked',x_masked.shape)
                       
        conv_list_spr = self.feature_extractor_spr(x_spr_m)  ####5 channels - for sparse2dense - 4 channels for image synth
                               
        if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):
            feature_spr = self.slicing_module_spr(conv_list_spr, conv_list_spr[0].shape[3]) ##### out_w = x_spr.shape[3] : force interpolation to max_w
                             
               
        if(self.compression_type == 'mhsa' or self.compression_type == 'mhsa_no_inter'):
            ##print('using mhsa')
            feature_spr = feature_spr.permute(0, 2, 1)  # [b, w, c*h]
            output_spr = self.mhsa_spr(feature_spr)                                 
            output_spr = self.drop_out(output_spr)
            output_spr = output_spr.permute(0, 2, 1) ###restore order
        else:
            ##print('using no mhsa')
            output_spr = feature_spr 
                
        output_spr = output_spr.reshape(output_spr.shape[0], output_spr.shape[1], 1, output_spr.shape[2])
                  

        for i in range(len(self.decoder1)): 
            ##
            if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):            
                output_spr = F.interpolate(output_spr, scale_factor=self.s_lst[i], mode='nearest')  
            else:
                output_spr = F.interpolate(output_spr, scale_factor=self.stride, mode='nearest')
            
            output_spr = self.decoder1[i](output_spr) 
            
        if(self.ref_type == 'rgbd'):
            output_spr = F.interpolate(output_spr, scale_factor=self.s_lst[i], mode='nearest') ####Bx4xhxw

            out_rgb = self.decoder_rgb(output_spr)
            out_d = self.decoder_d(output_spr)
            
            out_rgb = torch.clamp(out_rgb, 0, 1)

            ##print(out_rgb.shape,out_d.shape)
                            
            output_spr = torch.cat([out_rgb,out_d],dim=1)
           
                                                 
        output_spr = output_spr.squeeze(1)                 
                                                                           
        return output_spr

class FastSliceNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, full_size = True, sparse_encoder = False, compression_type='mhsa', ref_type = 'none', max_size = 1024, freeze_layout = False):
        super(FastSliceNet, self).__init__()
        self.backbone = backbone
        self.ch_scale = 4  
        
        self.ref_type = ref_type

        print('using ref', self.ref_type)########TO DO set last decoder stage
                
        self.gated_channels = 3 ####only rgb - no gating

        #if(self.ref_type == 'mask_one' or self.ref_type == 'rgbs2rgbd'):
        #    self.gated_channels = 4

        #if(self.ref_type == 'rgbd'):
        #    self.gated_channels = 3
        
        self.full_size = full_size 

        self.freeze_layout = freeze_layout

        if(not self.full_size):
            max_size = 512
              
        h,w = max_size//2,max_size
        self.max_enc = max_size                
        
        self.compression_type = compression_type ####NB. sed for sparse feature refinement

        print('using compression', self.compression_type)
        
        self.sparse_encoder = sparse_encoder

        #####features eencoder block
        
        print('using single branch rgbs encoder')

        #if(self.backbone == 'resnet18'):
        #    self.feature_extractor_spr = resnet18_channels(input_channels=self.gated_channels) #### - set 3 channels
            #
        
        if(self.sparse_encoder):
            self.gated_channels = 4
            if(self.backbone == 'resnet18'):
                self.feature_extractor_spr = resnet18_rgbs_channel_gated(inplanes = 64, input_channels=self.gated_channels) #### - set 4 channels + mask

            if(self.backbone == 'resnet34'):
                self.feature_extractor_spr = resnet34_rgbs_channel_gated(inplanes = 64, input_channels=self.gated_channels) #### 
        else:
            self.feature_extractor_spr = resnet18_channels(input_channels=self.gated_channels) #### - set 3 channels

        if(self.ref_type == 'depth_layout'):
            self.d2l = D2L(gpu=True, H = h, W = w)
            self.layout_from_depth = SegNet(backbone='light_depth')

            ###TO DO set pretrained
            print("Freezing segmentation network's weights\n")
            
            if(self.freeze_layout):
                for param in self.layout_from_depth.parameters():
                    param.requires_grad = False

                       
        # Inference channels number from each block of the encoder
        with torch.no_grad():
            ##self.lfeats *= (w//512)#####FIXME - hires support
            c1, c2, c3, c4 = 64, 128, 256, 512
            self.lfeats = w

            ##print('latent feat',self.lfeats)
        ##########################################
        
        #####features compression block###########################
                
        self.stride=(2,1)
        self.slicing_module_spr = MultiSlicing(c1, c2, c3, c4, self.ch_scale, stride=self.stride)###NB. not using gating in slicing, gated_conv = self.sparse_gated_encoder)
                                            
        ##################################################################
         
        #####feature refinement block##################################################
                
        if(self.compression_type == 'mhsa_no_inter'):
            print('using mhsa no inter for psarse branch')
            ####default layers=1
            self.mhsa_spr = MHSATransformerPos(num_layers=4, d_model=self.lfeats, num_heads=4, conv_hidden_dim=self.lfeats//2, maximum_position_encoding = self.max_enc//4)
        ################################################################################

        self.drop_out = nn.Dropout(0.5)        

        self.decoder1 = nn.ModuleList([])
                                
        #if(full_size):
        dec_count = int(math.log2(w))-1

        print('dec count',dec_count)

        last_ch = 1         
        
        print('only depth mode')
        for i in range(dec_count):
            #                
            sf1 = pow(2,i)
                                                               
            sf2 = 2*sf1
                                
            ch_in  = self.lfeats // sf1
            ch_out = self.lfeats // sf2

            if(i == dec_count-1):
                ch_out = 1                    

            ##print('scale ch', ch_in, ch_out)                                
                                
            decoder1 = nn.Sequential(
                        AConv(ch_in, ch_out, st=(1, 1)), ##gated_conv = self.sparse_gated_encoder),                            
                    )                
                                
            self.decoder1.append(decoder1)
            
        if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):
            self.s_lst = []
            for i in range(dec_count+1):
                ups = (2,1)
                if(i<2):
                    ups = (2,2)
                self.s_lst.append(ups)              
                                              
        ''' Pad left/right-most to each other instead of zero padding '''       
        ####FIXME - use only for rgb modules - gated sparse already have
        wrap_lr_pad(self)           
     
    def _prepare_rgb(self, x):
        ####CHECK IT for sparse input
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
                    
        return (x[:, :3] - self.x_mean) / self.x_std
            
    def forward(self, x_rgb, x_spr = None, mask_depth = False):
        x_rgb = self._prepare_rgb(x_rgb)
        
        if(self.sparse_encoder):
            b, c, h, w = x_rgb.size() 

            if(x_spr == None):
                x_spr = torch.zeros(b, 1, h, w).to(x_rgb.device)###

            x_rgb = torch.cat((x_rgb, x_spr), dim=1)
                       
        conv_list_spr = self.feature_extractor_spr(x_rgb)  ####3 channels    
        
        ##print(conv_list_spr[0].shape,conv_list_spr[3].shape)
                
        if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):
            feature_spr = self.slicing_module_spr(conv_list_spr, conv_list_spr[0].shape[3]) ##### out_w = x_spr.shape[3] : force interpolation to max_w

            ##print('feature_sparse',feature_spr.shape)
        if(self.compression_type == 'none'): ####FIXME - for back compatibilty
            feature_spr = self.slicing_module_spr(conv_list_spr, x_rgb.shape[3]) ##### out_w = x_spr.shape[3] : force interpolation to max_w
            ##print('feature_sparse',feature_spr.shape)
            
        ##print('latent shape',feature_spr.shape)
               
        if(self.compression_type == 'mhsa' or self.compression_type == 'mhsa_no_inter'):
            ##print('using mhsa')
            feature_spr = feature_spr.permute(0, 2, 1)  # [b, w, c*h]
            output_spr = self.mhsa_spr(feature_spr)                                 
            output_spr = self.drop_out(output_spr)
            output_spr = output_spr.permute(0, 2, 1) ###restore order
        else:
            ##print('using no mhsa')
            output_spr = feature_spr 
                            
        output_spr = output_spr.reshape(output_spr.shape[0], output_spr.shape[1], 1, output_spr.shape[2])
                  

        for i in range(len(self.decoder1)): 
            if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):            
                output_spr = F.interpolate(output_spr, scale_factor=self.s_lst[i], mode='nearest')  
            else:
                output_spr = F.interpolate(output_spr, scale_factor=self.stride, mode='nearest')
            
            output_spr = self.decoder1[i](output_spr)                  
         
        result = []
                
        if(mask_depth):
            print('using mask')
            x_mask = torch.where( (output_spr> 0.15) & (output_spr < 8.0) , 1.0, 0.).float()##(output_spr>0.1).float()
            output_spr = x_mask*output_spr

        result.append(output_spr.squeeze(1))####TO Bxhxw
                
        if(self.ref_type == 'depth_layout'):
            ###TO DO from output_spr to atl - batched
            ##print('input depth', output_spr.shape, output_spr.device)
            x_atl_depth, d_down, c_dist, f_dist, fl = self.d2l.batched_atlanta_transform_from_depth(output_spr)#####input Bx1xhxw
            ##print('output atl depth', x_atl_depth.shape, x_atl_depth.device)
            layout = self.layout_from_depth(x_atl_depth) ### input: Bx1xhxh
            result.append(layout)        
                                                                   
        return   result 

    def get_latent_feature(self, x_rgb):
        x_rgb = self._prepare_rgb(x_rgb)
        
        if(self.sparse_encoder):
            b, c, h, w = x_rgb.size() 
            x_spr = torch.zeros(b, 1, h, w).to(x_rgb.device)###

            x_rgb = torch.cat((x_rgb, x_spr), dim=1)

                       
        conv_list_spr = self.feature_extractor_spr(x_rgb)  ####3 channels        
                
        if(self.compression_type == 'mhsa_no_inter' or self.compression_type == 'no_inter'):
            feature_spr = self.slicing_module_spr(conv_list_spr, conv_list_spr[0].shape[3]) ##### out_w = x_spr.shape[3] : force interpolation to max_w

            ##print('feature_sparse',feature_spr.shape)
        if(self.compression_type == 'none'): ####FIXME - for back compatibilty
            feature_spr = self.slicing_module_spr(conv_list_spr, x_rgb.shape[3]) ##### out_w = x_spr.shape[3] : force interpolation to max_w
            ##print('feature_sparse',feature_spr.shape)
                 
                                                                          
        return  feature_spr  


def counter_rgb():
    print('testing gflops')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' ####FIXMEEEE

    device = torch.device('cuda')

    test_full = True

    ##h,w =  256,512##512,1024
    h,w =  512,1024
    ##h,w =  1024,2048##512,1024
    ##h,w =  2048,4096#
    ##h,w =  4096,8192#

    ##net = FastSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'no_inter', ref_type = 'depth_layout', max_size = w).to(device)####NB. using this
    net = FastSliceNet('resnet18',full_size = test_full, sparse_encoder = True, compression_type = 'no_inter', ref_type = 'depth_layout', max_size = w).to(device)
    ##net = GatedSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'mhsa_no_inter', ref_type = 'mask_one', max_size = w).to(device)
    ##net = GatedSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'no_inter', ref_type = 'mask_one', max_size = w).to(device)####NB. using this configuration
    ##net = GatedSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'no_inter', ref_type = 'rgbd', max_size = w).to(device)
    ##net = GatedNet(device, decoder_type='rgbd', backbone ='light').to(device)
    
    # testing
    layers = net
   

    if(test_full):
        inputs = []
        img = torch.randn(1, 3, h, w).to(device)
        inputs.append(img)
        #mask = torch.randn(1, 1, h, w).to(device)
        #inputs.append(mask)
        #m_img = torch.randn(1, 3, h, w).to(device)
        #inputs.append(m_img)
    else:
        inputs = []
        img = torch.randn(1, 3, 256, 512).to(device)
        inputs.append(img)
        ##inputs.append(device)

    ##out = layers(img,mask,masked_input,device)

    with torch.no_grad():
        flops, params = profile(layers, inputs)
    ##print(f'input :', [v.shape for v in inputs])
    print(f'flops : {flops/(10**9):.2f} G')
    print(f'params: {params/(10**6):.2f} M')

    import time
    fps = []
    with torch.no_grad():
        out, layout = layers(img)

        print(out.shape, layout.shape)

        ##lf = net.get_latent_feature(img,mask,m_img)
        ##print('latent',lf.shape)

                
        for _ in range(50):
            eps_time = time.time()
            layers(img)
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
    print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')

def counter_rgbs():
    print('testing gflops')

    ##os.environ['CUDA_VISIBLE_DEVICES'] = '0' ####FIXMEEEE

    device = torch.device('cuda')
     

    test_full = True

    if(test_full):
        h,w =  512,1024
        ##h,w =  1024,2048##512,1024
        ##h,w =  2048,4096#
        ##h,w =  4096,8192#
    else:
        h,w =  256,512##512,1024

    net = FastSliceNet('resnet18',full_size = test_full, sparse_encoder = True, compression_type = 'mhsa_no_inter', ref_type = 'depth_layout',max_size = w).to(device)
    ##net = GatedSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'mhsa_no_inter', ref_type = 'mask_one', max_size = w).to(device)
    ##net = GatedSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'no_inter', ref_type = 'mask_one', max_size = w).to(device)
    ##net = GatedSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'no_inter', ref_type = 'rgbd', max_size = w).to(device)
    ##net = GatedNet(device, decoder_type='rgbd', backbone ='light').to(device)
    
    # testing
    layers = net  
        
    inputs = []
    img = torch.randn(1, 3, h, w).to(device)
    inputs.append(img)
    mask = torch.randn(1, 1, h, w).to(device)
    inputs.append(mask)
             
    ##inputs.append(device)

    ##out = layers(img,mask,masked_input,device)

    with torch.no_grad():
        flops, params = profile(layers, inputs)
    ##print(f'input :', [v.shape for v in inputs])
    print(f'flops : {flops/(10**9):.2f} G')
    print(f'params: {params/(10**6):.2f} M')

    import time
    fps = []
    with torch.no_grad():
        out,_ = layers(img,mask)

        print(out.shape)

        ##lf = net.get_latent_feature(img,mask,m_img)
        ##print('latent',lf.shape)

                
        for _ in range(50):
            eps_time = time.time()
            layers(img,mask)
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
    print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')


def counter_slicenet():
    print('testing gflops')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' ####FIXMEEEE

    device = torch.device('cuda')

    test_full = True

    ##h,w =  256,512##512,1024
    h,w =  512,1024
    ##h,w =  1024,2048##512,1024
    ##h,w =  2048,4096#
    ##h,w =  4096,8192#

    ##net = GatedSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'mhsa_no_inter', ref_type = 'mask_one', max_size = w).to(device)
    ##net = GatedSliceNet('resnet18',full_size = test_full, sparse_encoder = False, compression_type = 'no_inter', ref_type = 'mask_one', max_size = w).to(device)
    net = SliceNet('resnet18_rgbd',full_size = test_full, sparse_encoder = False, compression_type = 'no_inter', ref_type = 'none', max_size = w).to(device)
    ##net = GatedNet(device, decoder_type='rgbd', backbone ='light').to(device)
    
    # testing
    layers = net
   

    if(test_full):
        inputs = []
        img = torch.randn(1, 3, h, w).to(device)
        inputs.append(img)
        d = torch.randn(1, 1, h, w).to(device)
        inputs.append(d)
       
    else:
        inputs = []
        img = torch.randn(1, 3, 256, 512).to(device)
        inputs.append(img)
        d = torch.randn(1, 1, 256, 512).to(device)
        inputs.append(d)
        
        
        ##inputs.append(device)

    ##out = layers(img,mask,masked_input,device)

    with torch.no_grad():
        flops, params = profile(layers, inputs)
    ##print(f'input :', [v.shape for v in inputs])
    print(f'flops : {flops/(10**9):.2f} G')
    print(f'params: {params/(10**6):.2f} M')

    import time
    fps = []
    with torch.no_grad():
        out = layers(img,d)

        print('out',out.shape)

        ##lf = net.get_latent_feature(img,mask,m_img)
        ##print('latent',lf.shape)

                
        for _ in range(50):
            eps_time = time.time()
            layers(img,d)
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
    print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')


if __name__ == '__main__':
    ##counter()
    ##counter_rgbs()
    ##counter_rgbs_tr()
    ##counter_slicenet()
    counter_rgbs()
