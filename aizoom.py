import sys
sys.path.append("./lib")
from lib.upsampler import Upsampler
import argparse
import sys

parser = argparse.ArgumentParser(
    description="zoom an image 2X o 4X via AI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('fin', metavar='FNAME',
                    type=str, help='Input image')
parser.add_argument('-o', '--output', dest='fout',
                    required=False, default='result.png', help="Output image")
parser.add_argument('--zoom-factor', dest='zf', required=False, default=2, type=int, help="Zoom factor")
args = parser.parse_args()

fin = args.fin
fout = args.fout
zf = args.zf

if zf != 1 and zf != 2 and zf != 4:
    sys.exit("zoom_factor must be 1, 2 or 4!")

upsampler = Upsampler(args.zf)
upsampler.infer(fin, fout)
