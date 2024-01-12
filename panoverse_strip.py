import sys
sys.path.append("./lib")
import os
import glob
# import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import math
import torch
import time
import json

from indoor_synth_model import IndoorSynthNet
from slicenet_gated_model_scalable import GatedSliceNet

from misc import tools
from misc.layout import *

import numpy.ma as ma
from scipy.interpolate import interp2d


def depth_inference(img_path, depth_path):

    device = torch.device('cuda')

    # choosing depth estimator
    use_gatednet = True  # NB. PanoVerse case

    if (use_gatednet):
        d_pth = 'ckpt/S3D_gatednet_combo_depth_atl_layout_gated_max_min2/best_valid.pth'
        dl_net = tools.load_combo_trained_model(
            IndoorSynthNet, device, d_pth).to(device)
        dl_net.eval()
    else:
        d_pth = 'ckpt/S3D_combo_depth_atl_layout_gated_max_min_loss_D_comp2/best_valid.pth'
        dl_net = tools.load_trained_model(FastSliceNet, d_pth).to(device)
        dl_net.eval()

    # load source image
    img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
    x_img = torch.FloatTensor(img.transpose([2, 0, 1]).copy())

    # STEP 1 inference: depth
    with torch.no_grad():
        # basic input
        x_img = x_img.unsqueeze(0)  # to batch - NB. same for all predictions
        x_depth, _ = dl_net(x_img.to(device))
        x_depth *= 1000.0  # to mm
        Image.fromarray(x_depth.squeeze(
            0).cpu().numpy().astype(np.uint16)).save(depth_path)


def view_inference(img_path, depth_path, outimg_path, hr, angle, padded=False):
    device = torch.device('cuda')
    # load source image
    img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
    x_img = torch.FloatTensor(img.transpose([2, 0, 1]).copy()).unsqueeze(0)

    # unsigned 16-bit integers within a PNG. The units are millimeters
    depth = np.array(Image.open(depth_path), np.float32)
    depth /= 1000.0  # to meters

    x_depth = torch.FloatTensor(depth).unsqueeze(0)

    # Loading view synth model trained model
    if (angle != None):
        pth = 'ckpt/gatednet_pnvs_full_dr_sliced160_no_adv/best_valid.pth'
    else:
        pth = 'ckpt/gatednet_pnvs_full_rgb_gt_depth_layout_edges_adv_0_shot_dr/best_valid.pth'

    net = tools.load_emptying_room_trained_model(
        IndoorSynthNet, device, pth).to(device)
    net.eval()
    net.get_mapping = False

    with torch.no_grad():

        # STEP 2 inference: translated views
        x_ts = torch.FloatTensor(np.zeros((1, 1, 3))).to(
            device)  # init translation to 0

        x_ts[:, :, 0] = hr*math.sin(float(angle))
        x_ts[:, :, 1] = hr*math.cos(float(angle))

        slice = 512  # default image center

        if (angle != None):
            slice = int((float(x_img.shape[3])/(2*math.pi))
                        * angle + (x_img.shape[3]//2))
            print('generating sample for angle', float(angle), slice)

            rgbd_sample, _ = net(x_img.to(device), x_ts.to(
                device), x_depth.to(device), slice_c=slice)
        else:
            rgbd_sample, _ = net(x_img.to(device), x_ts.to(
                device), x_depth.to(device))

        if (padded):
            pred_pad = torch.zeros_like(x_img)
            shift_i = int(slice - x_img.shape[3]//2)  # rotate from origin
            pred_pad = torch.roll(pred_pad, (0, -shift_i),
                                  dims=(2, 3))  # neg - clockwise
            pos = int((x_img.shape[3]//2)-(net.slice_w // 2))
            pred_pad[:, :, 0:x_img.shape[2], pos:net.slice_w+pos] = rgbd_sample
            pred_pad = torch.roll(pred_pad, (0, shift_i), dims=(2, 3))
            rgb_sample = tools.x2image(pred_pad.cpu().squeeze(0))
            im1 = Image.fromarray(rgb_sample)
            im1.save(outimg_path)
        else:
            rgb_sample = tools.x2image(rgbd_sample.cpu().squeeze(0))
            im1 = Image.fromarray(rgb_sample)
            im1.save(outimg_path)

        # match_file1 = save_path + os.path.basename(img_path)+'_'+str(slice)+'.txt'
        # np.savetxt(match_file1, matches.squeeze(0).cpu().numpy().astype(np.int32), fmt='%s %s %s %s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('imgpath', metavar='FNAME',
                        type=str, help='Ground Truth Image')
    parser.add_argument('-o', '--output', dest='outdir',
                        required=False, default='result', help="Output directory")
    parser.add_argument('--samples', dest='samples', required=False,
                        default=16, type=int, help="Number of samples")
    parser.add_argument('--head-radius', dest='head_radius', required=False, help='Radius of the head',
                        default=0.100, type=float)
    parser.add_argument('--ipd', dest='ipd', required=False, help='IPD',
                        default=0.064, type=float)
    parser.add_argument('--padded', dest='padded',
                        help="", default=False, action='store_true')
    args = parser.parse_args()

    # Check target directory
    if not os.path.isdir(args.outdir):
        print('Output directory %s not existed. Create one.' % args.outdir)
        os.makedirs(args.outdir)

    basename = os.path.basename(args.imgpath)
    scenename, ext = os.path.splitext(basename)
    depthname = f"{args.outdir}/{scenename}_depth.png"
    jsonname = f"{args.outdir}/{scenename}.json"
    config = {"head_radius": args.head_radius,
              "ipd": args.ipd,
              "data": []}

    print(f"Basename: {basename}")
    print(f"Scenename: {scenename}")
    print(f"Depthname: {depthname}")

    print(f"jsonname: {jsonname}")

    depth_inference(img_path=args.imgpath,
                    depth_path=depthname)

    step_deg = 360 / float(args.samples)

    for ids in range(args.samples):
        id_label = f'{ids:04}'
        outname = f"{args.outdir}/{scenename}_{id_label}.png"
        print(f"outname: {outname}")

        theta_deg = float(ids*step_deg)-180
        theta = math.radians(theta_deg)

        config["data"].append({"theta": theta_deg, "keyimg": outname})
        view_inference(img_path=args.imgpath, depth_path=depthname, outimg_path=outname,
                       hr=args.head_radius, angle=theta, padded=args.padded)

config['data'] = sorted(config['data'], key=lambda x: x['theta'])
with open(jsonname, 'w') as file_json:
    json.dump(config, file_json, indent=4)
