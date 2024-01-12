import argparse
import numpy as np
from PIL import Image
import json
import math
import os

def get_parent_directory(pathname):
    # Use os.path.split to get the directory and the last component of the path
    directory, _ = os.path.split(pathname)
    return directory

def relative_pathname(pathname, prepend=''):
    # Use os.path.relpath to make the pathname relative to the current working directory
    path = pathname if prepend == '' else f"{prepend}/{pathname}"
    relative_pathname = os.path.relpath(path)
    return relative_pathname

def sceneName(s):
    result = None
    parts = s.split("_")
    if len(parts) >= 2:
        result = '_'.join(parts[:-1])
    return result


def gaussian(x, sigma):
    return np.exp(-0.5 * ((x / sigma) ** 2))


def alpha_blend_images(image1, image2, image3, mi, sigma):
    height, width, _ = image1.shape
    blended_image = np.zeros_like(image1, dtype=np.float32)

    for x in range(width):
        alpha = gaussian(x - mi, sigma)
        if (x <= mi):
            blended_image[:, x, :] = (1 - alpha) * \
                image1[:, x, :] + alpha * image2[:, x, :]
        else:
            blended_image[:, x, :] = alpha * \
                image2[:, x, :] + (1-alpha) * image3[:, x, :]

    return blended_image.astype(np.uint8)


def sliceMiddle(image, sliceWidth):
    height, width, _ = image.shape
    start_x = int((width - sliceWidth) // 2)
    end_x = int(start_x + sliceWidth)
    print(f"{start_x}:{end_x}")
    return image[:, start_x:end_x, :]


def getSlice(parentdir, data, idxs, sliceWidth, mi, blending):
    icurr = idxs['icurr']
    iprev = idxs['iprev']
    inext = idxs['inext']
    print(f"KEYIMG {data[icurr]['keyimg']}")
    image1 = Image.open(relative_pathname(data[iprev]['keyimg'], parentdir))
    image2 = Image.open(relative_pathname(data[icurr]['keyimg'], parentdir))
    image3 = Image.open(relative_pathname(data[inext]['keyimg'], parentdir))
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    arr3 = np.array(image3)
    height, width, _ = arr1.shape
    rollx = int(width/2 - mi)
    arr1 = np.roll(arr1, rollx, axis=1)
    arr2 = np.roll(arr2, rollx, axis=1)
    arr3 = np.roll(arr3, rollx, axis=1)
    sigma = sliceWidth/4
    out = alpha_blend_images(arr1, arr2, arr3, mi, sigma) if blending else arr2
    return sliceMiddle(out, sliceWidth)


def copyAt(xc, slice, imgOut):
    height, width, _ = imgOut.shape
    sliceHeight, sliceWidth, _ = slice.shape
    xc = (xc + width) % width
    d0 = int(xc - sliceWidth * 0.5)
    d1 = int(xc + sliceWidth * 0.5)
    print(f"d0 = {d0}    d1 = {d1}")
    if d0 < 0:
        imgOut[:, d0+width: width, :] = slice[:, :-d0, :]
        imgOut[:, :d1, :] = slice[:, -d0:, :]
    elif d1 > width:
        imgOut[:, :d1-width, :] = slice[:, -(d1-width):, :]
        imgOut[:, d0: width, :] = slice[:, :(width-d0), :]
    else:
        imgOut[:, d0:d1, :] = slice


def find_closest(data, theta):
    # Normalize theta to the range [-180, 180]
    while theta < -180:
        theta += 360
    while theta >= 180:
        theta -= 360
    min_idx = 0
    min_dist = abs(data[0]['theta'] - theta)
    n = len(data)
    for i in range(1, n):
        dist = abs(data[i]['theta'] - theta)
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    icurr = min_idx
    iprev = icurr - 1 if icurr > 0 else n-1
    inext = (icurr + 1) % n
    return {
        'icurr': icurr,
        'iprev': iprev,
        'inext': inext
    }


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Convert a set of equirectangular images into two stereoscopic equirectangural images')
parser.add_argument('fjson', metavar='FNAME',
                    type=str, help='JSON config')
parser.add_argument('-o', '--output', dest='dirout',
                    required=False, default='/tmp', help="Output directory")
parser.add_argument('--blending', dest='blending',
                        help="", default=False, action='store_true')
args = parser.parse_args()

fjson = args.fjson
fjsondir = os.path.dirname(fjson)
fjsonparentdir = get_parent_directory(fjsondir)

print("DIRS")
print(fjsondir, fjsonparentdir)

dirout = args.dirout
# Check target directory
if not os.path.isdir(dirout):
    print('Output directory %s not existed. Create one.' % dirout)
    os.makedirs(dirout)

with open(fjson) as f:
    config = json.load(f)

data = config['data']
data = sorted(data, key=lambda x: x['theta'])
print(f"SORTED ", data)


ipd = config['ipd']
head_radius = config['head_radius']

n = len(data)
dangle = 360 / n
print(f"{data[0]['keyimg']}")

fimg0 = data[0]['keyimg']
basename0 = os.path.basename(fimg0)
name0, ext0 = os.path.splitext(basename0)
scenename = sceneName(name0)

print(f"REL: {relative_pathname(fimg0, fjsonparentdir)}")
img0 = Image.open(relative_pathname(fimg0, fjsonparentdir))
arr0 = np.array(img0)
width, height = img0.size
slice_width = round(width / n)
if slice_width % 2 != 0:
    slice_width += 1
dtheta = math.degrees(math.atan2(ipd*0.5, head_radius))
#dtheta = 90
sigma = slice_width / 4

out_l = np.zeros_like(arr0)
out_r = np.zeros_like(arr0)
fout_l = f"{dirout}/{scenename}_l.png"
fout_r = f"{dirout}/{scenename}_r.png"

print("#### INFO")
print(f"Scene Name: {scenename}")
print(f"head_radius: {head_radius}  IPD: {ipd}")
print(f"n: {n}   delta_angle: {dangle}  img_size: {width}X{height}")
print(f"slice_width: {slice_width}  delta_eyes: {dtheta}")
print("###################")

for i in range(n):
    theta = (dangle * i - 180) # [-180, 180]
    x_c = int((theta+180)/360 * width)
    idxs = find_closest(data, theta)
    print(f"C {theta} {idxs}")

    theta_l = theta - dtheta
    x_l = int((theta_l+180)/360 * width)
    idxs_l = find_closest(data, theta_l)
    slice_l = getSlice(fjsonparentdir, data, idxs_l, slice_width, x_c, args.blending)
    copyAt(x_c, slice_l, out_l)
    print(f"L {theta_l} {idxs_l}")

    theta_r = theta + dtheta
    x_r = int((theta_r+180)/360 * width)
    idxs_r = find_closest(data, theta_r)
    slice_r = getSlice(fjsonparentdir, data, idxs_r, slice_width, x_c, args.blending)
    copyAt(x_c, slice_r, out_r)
    print(f"R {theta_r} {idxs_r}\n")

img_l = Image.fromarray(out_l)
img_r = Image.fromarray(out_r)
img_l.save(fout_l)
img_r.save(fout_r)
