import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SquaredGradientLoss(nn.Module):
	'''Compute the gradient magnitude of an image using the simple filters as in: 
	Garg, Ravi, et al. "Unsupervised cnn for single view depth estimation: Geometry to the rescue." European Conference on Computer Vision. Springer, Cham, 2016.
	'''

	def __init__(self):

		super(SquaredGradientLoss, self).__init__()

		self.register_buffer('dx_filter', torch.FloatTensor([
				[0,0,0],
				[-0.5,0,0.5],
				[0,0,0]]).view(1,1,3,3))
		self.register_buffer('dy_filter', torch.FloatTensor([
				[0,-0.5,0],
				[0,0,0],
				[0,0.5,0]]).view(1,1,3,3))

	def forward(self, pred, mask):
		dx = F.conv2d(
			pred, 
			self.dx_filter.to(pred.get_device()), 
			padding=1, 
			groups=pred.shape[1])
		dy = F.conv2d(
			pred, 
			self.dy_filter.to(pred.get_device()), 
			padding=1, 
			groups=pred.shape[1])

		error = mask * \
			(dx.abs().sum(1, keepdim=True) + dy.abs().sum(1, keepdim=True))

		return error.sum() / (mask > 0).sum().float()

def laplacian(img):
    img = img.unsqueeze(1)

    f_xy = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(f_xy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    
    return conv1(img)

def smoothness_map(img, order = 2):
    img = img.unsqueeze(1)
    #####from Surface Reconstruction with Higher-Order Smoothness
    if(order == 2):
        f_xx = np.array([[0,0,0],[1,-2,1],[0,0,0]])
        f_yy = np.array([[0,1,0],[0,-2,0],[0,1,0]])
        f_xy = np.array([[1/4,0,-1/4],[0,0,0],[-1/4,0,1/4]])

        pad = 1
    else:
        ####NB only order 4 is supported here
        f_xx = np.array([[0,0,0,0,0],[0,0,0,0,0],[-1/12,4/3,-5/2,4/3,-1/12],[0,0,0,0,0],[0,0,0,0,0]])
        f_yy = np.array([[0,0,-1/12,0,0],[0,0,4/3,0,0],[0,0,-5/2,0,0],[0,0,4/3,0,0],[0,0,-1/12,0,0]])
        f_xy = np.array([[1/24,-1/18,0,1/18,-1/24],[-1/18,4/9,0,-4/9,1/18],[0,0,0,0,0],[1/18,-4/9,0,4/9,-1/18],[-1/24,1/18,0,-1/18,1/24]])

        pad = 2


    conv_xx = nn.Conv2d(1, 1, kernel_size=order+1, stride=1, padding=pad, bias=False)
    weight = torch.from_numpy(f_xx).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()
    conv_xx.weight = nn.Parameter(weight)

    diff_xx = conv_xx(img)

    conv_yy = nn.Conv2d(1, 1, kernel_size=order+1, stride=1, padding=pad, bias=False)
    weight = torch.from_numpy(f_yy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()
    conv_yy.weight = nn.Parameter(weight)

    diff_yy = conv_yy(img)

    conv_xy = nn.Conv2d(1, 1, kernel_size=order+1, stride=1, padding=pad, bias=False)
    weight = torch.from_numpy(f_xy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()
    conv_xy.weight = nn.Parameter(weight)

    diff_xy = conv_xy(img)

    s_xx = diff_xx * diff_xx
    s_yy = diff_yy * diff_yy
    s_xy = diff_xy * diff_xy

    smoothness = s_xx + s_yy + 2 * s_xy

    smoothness = F.normalize(smoothness, dim=2)

    eps = 10e-3 ###takes in account order apporx - TO DO

    smoothness[(abs(smoothness)<eps)]= 0
    
    return smoothness  

def curvature_map(img, order = 2):
    img = img.unsqueeze(1)
    #####from Surface Reconstruction with Higher-Order Smoothness

    if(order == 2):
        f_xx = np.array([[0,0,0],[1,-2,1],[0,0,0]])
        f_yy = np.array([[0,1,0],[0,-2,0],[0,1,0]])
        f_xy = np.array([[1/4,0,-1/4],[0,0,0],[-1/4,0,1/4]])

        pad = 1
    else:
        ####NB only order 4 is supported here
        f_xx = np.array([[0,0,0,0,0],[0,0,0,0,0],[-1/12,4/3,-5/2,4/3,-1/12],[0,0,0,0,0],[0,0,0,0,0]])
        f_yy = np.array([[0,0,-1/12,0,0],[0,0,4/3,0,0],[0,0,-5/2,0,0],[0,0,4/3,0,0],[0,0,-1/12,0,0]])
        f_xy = np.array([[1/24,-1/18,0,1/18,-1/24],[-1/18,4/9,0,-4/9,1/18],[0,0,0,0,0],[1/18,-4/9,0,4/9,-1/18],[-1/24,1/18,0,-1/18,1/24]])

        pad = 2


    conv_xx = nn.Conv2d(1, 1, kernel_size=order+1, stride=1, padding=pad, bias=False)
    weight = torch.from_numpy(f_xx).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()
    conv_xx.weight = nn.Parameter(weight)

    diff_xx = conv_xx(img)

    conv_yy = nn.Conv2d(1, 1, kernel_size=order+1, stride=1, padding=pad, bias=False)
    weight = torch.from_numpy(f_yy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()
    conv_yy.weight = nn.Parameter(weight)

    diff_yy = conv_yy(img)

    conv_xy = nn.Conv2d(1, 1, kernel_size=order+1, stride=1, padding=pad, bias=False)
    weight = torch.from_numpy(f_xy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()
    conv_xy.weight = nn.Parameter(weight)

    diff_xy = conv_xy(img)

    s_xx = diff_xx * diff_xx
    s_yy = diff_yy * diff_yy
    s_xy = diff_xy * diff_xy

    curv = s_xx * s_yy - 2 * s_xy

    curv = F.normalize(curv, dim=2)

    eps = 10e-5 ###takes in account order apporx - TO DO

    curv[(abs(curv)<eps)]= 0
    
    return curv    

class L2Loss(nn.Module):

	def __init__(self):

		super(L2Loss, self).__init__()

		self.metric = nn.MSELoss()

	def forward(self, pred, gt, mask):
		error = mask * self.metric(pred, gt)
		return error.sum() / (mask > 0).sum().float()


class MultiScaleL2Loss(nn.Module):

	def __init__(self, alpha_list, beta_list):

		super(MultiScaleL2Loss, self).__init__()

		self.depth_metric = L2Loss()
		self.grad_metric = SquaredGradientLoss()
		self.alpha_list = alpha_list
		self.beta_list = beta_list

	def forward(self, pred_list, gt_list, mask_list):

		# Go through each scale and accumulate errors
		depth_error = 0
		for i in range(len(pred_list)):

			depth_pred = pred_list[i]
			depth_gt = gt_list[i]
			mask = mask_list[i]
			alpha = self.alpha_list[i]
			beta = self.beta_list[i]

			# Compute depth error at this scale
			depth_error += alpha * self.depth_metric(
				depth_pred, 
				depth_gt, 
				mask)
		
			# Compute gradient error at this scale
			depth_error += beta * self.grad_metric(
				depth_pred, 
				mask)
		
		return depth_error

##new 
def inverse_huber_loss(target,output):
    absdiff = torch.abs(output-target)
    C = 0.2*torch.max(absdiff).item()
    return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))


class inverse_huber_loss_with_mask(nn.Module):

	def __init__(self):

		super(inverse_huber_loss_with_mask, self).__init__()

		self.metric = nn.MSELoss()

	def forward(self, pred, gt, mask):        
		error = mask * inverse_huber_loss(pred, gt)
		return error.sum() / (mask > 0).sum().float()



