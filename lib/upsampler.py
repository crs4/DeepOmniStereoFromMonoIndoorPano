from basicsr.archs.rrdbnet_arch import RRDBNet

import numpy as np
from PIL import Image
import torch
from realesrgan import RealESRGANer
import cv2 as cv

class Upsampler:
    def_sr2_pth='ckpt/RealESRGAN_x2plus.pth'
    def_sr4_pth='ckpt/RealESRGAN_x4plus.pth'
    def __init__(self, zoom_factor=2):
        self.zoom_factor = zoom_factor

        self.device = torch.device('cuda')

        def_sr_pth = self.def_sr2_pth if zoom_factor == 2 else self.def_sr4_pth

         # sr model
        self.sr_net = None
        self.upsampler = None

        if (self.zoom_factor == 2 or self.zoom_factor == 4):
            self.sr_net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=zoom_factor)
            self.upsampler = RealESRGANer(
                    scale=zoom_factor,
                    model_path=def_sr_pth,
                    dni_weight=None,
                    model=self.sr_net,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=True,
                    gpu_id=0)
        
    def infer(self, src_path, dst_path):
        src = cv.imread(src_path, cv.IMREAD_UNCHANGED)
        dst = src
        if (self.upsampler):
            dst, _ = self.upsampler.enhance(src, outscale=self.zoom_factor)
        cv.imwrite(dst_path, dst)

    def inferArr(self, src):
        dst = src
        if (self.upsampler):
            dst, _ = self.upsampler.enhance(src, outscale=self.zoom_factor)
        return dst

