import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
import os
import cv2
import glob
import copy
import numpy as np
import cupy as cp
import torch
import imageio
from PIL import Image
import scipy.ndimage
from skimage.feature import canny
from skimage.transform import integral
import torchvision.transforms.functional as F

from RAFT import utils
from RAFT import RAFT

import utils.region_fill as rf
from utils.Poisson_blend import Poisson_blend
from utils.Poisson_blend_img import Poisson_blend_img
from get_flowNN import get_flowNN
from get_flowNN_gradient import get_flowNN_gradient
from utils.common_utils import flow_edge
from spatial_inpaint import spatial_inpaint
from frame_inpaint import DeepFillv1
from edgeconnect.networks import EdgeGenerator_

import time

def find_minbbox(masks):
    # find the minimum bounding box of the holdmask
    minbbox_tl=[]            # top left point of the minimum bounding box
    minbbox_br=[]            # bottom right point of the minimum bounding box
    for i in range (0,len(masks)):
        non_zeros=cv2.findNonZero(np.array(masks[i]*255))
        min_rect=cv2.boundingRect(non_zeros)    

        # expand 10 pixels 
        x1=max(0,min_rect[0]-10)
        y1=max(0,min_rect[1]-10)
        x2=min(masks[i].shape[1],min_rect[0]+min_rect[2]+10)
        y2=min(masks[i].shape[0],min_rect[1]+min_rect[3]+10)
        
        minbbox_tl.append([x1,y1])
        minbbox_br.append([x2,y2])
    return minbbox_tl,minbbox_br
    

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def infer(args, EdgeGenerator, device, flow_img_gray, edge, mask):

    # Add a pytorch dataloader
    flow_img_gray_tensor = to_tensor(flow_img_gray)[None, :, :].float().to(device)
    edge_tensor = to_tensor(edge)[None, :, :].float().to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float64))[None, None, :, :].float().to(device)

    # Complete the edges
    edges_masked = (edge_tensor * (1 - mask_tensor))
    images_masked = (flow_img_gray_tensor * (1 - mask_tensor)) + mask_tensor
    inputs = torch.cat((images_masked, edges_masked, mask_tensor), dim=1)
    with torch.no_grad():
        edges_completed = EdgeGenerator(inputs) # in: [grayscale(1) + edge(1) + mask(1)]
    edges_completed = edges_completed * mask_tensor + edge_tensor * (1 - mask_tensor)
    edge_completed = edges_completed[0, 0].data.cpu().numpy()
    edge_completed[edge_completed < 0.5] = 0
    edge_completed[edge_completed >= 0.5] = 1

    return edge_completed


def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1)))

    return gradient_mask


def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def calculate_flow(args, model, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    # if os.path.isdir(os.path.join(args.outroot, 'flow', mode + '_flo')):
    #     for flow_name in sorted(glob.glob(os.path.join(args.outroot, 'flow', mode + '_flo', '*.flo'))):
    #         print("Loading {0}".format(flow_name), '\r', end='')
    #         flow = utils.frame_utils.readFlow(flow_name)
    #         Flow = np.concatenate((Flow, flow[..., None]), axis=-1)
    #     return Flow

    # create_dir(os.path.join(args.outroot, 'flow', mode + '_flo'))
    # create_dir(os.path.join(args.outroot, 'flow', mode + '_png'))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            # # Flow visualization.
            # flow_img = utils.flow_viz.flow_to_image(flow)
            # flow_img = Image.fromarray(flow_img)

            # # Saves the flow and flow_img.
            # flow_img.save(os.path.join(args.outroot, 'flow', mode + '_png', '%05d.png'%i))
            # utils.frame_utils.writeFlow(os.path.join(args.outroot, 'flow', mode + '_flo', '%05d.flo'%i), flow)

    return Flow



def complete_flow(args, corrFlow, flow_mask, mode, minbbox_tl, minbbox_br, edge=None):
    """Completes flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW, _, nFrame = corrFlow.shape

    # if os.path.isdir(os.path.join(args.outroot, 'flow_comp', mode + '_flo')):
    #     compFlow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    #     for flow_name in sorted(glob.glob(os.path.join(args.outroot, 'flow_comp', mode + '_flo', '*.flo'))):
    #         print("Loading {0}".format(flow_name), '\r', end='')
    #         flow = utils.frame_utils.readFlow(flow_name)
    #         compFlow = np.concatenate((compFlow, flow[..., None]), axis=-1)
    #     return compFlow

    # create_dir(os.path.join(args.outroot, 'flow_comp', mode + '_flo'))
    # create_dir(os.path.join(args.outroot, 'flow_comp', mode + '_png'))

    compFlow=corrFlow.copy()
    for i in range(nFrame):
        print("Completing {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow = corrFlow[:, :, :, i]
        if mode == 'forward':            
            flow_crop=flow[minbbox_tl[i][1]:minbbox_br[i][1],minbbox_tl[i][0]:minbbox_br[i][0],:] 
            flow_mask_img = flow_mask[:, :, i]
            flow_mask_img_crop=flow_mask_img[minbbox_tl[i][1]:minbbox_br[i][1],minbbox_tl[i][0]:minbbox_br[i][0]]
        else:
            flow_crop=flow[minbbox_tl[i+1][1]:minbbox_br[i+1][1],minbbox_tl[i+1][0]:minbbox_br[i+1][0],:] 
            flow_mask_img = flow_mask[:, :, i+1]
            flow_mask_img_crop=flow_mask_img[minbbox_tl[i+1][1]:minbbox_br[i+1][1],minbbox_tl[i+1][0]:minbbox_br[i+1][0]]
        
        # cv2.imwrite("./flow_mask_img_crop.png",flow_mask_img_crop*255)
        flow_mask_gradient_img = gradient_mask(flow_mask_img)
        flow_mask_gradient_img_crop = gradient_mask(flow_mask_img_crop)

        if edge is not None:
            # imgH x (imgW - 1 + 1) x 2
            gradient_x = np.concatenate((np.diff(flow_crop, axis=1), np.zeros((flow_crop.shape[0], 1, 2), dtype=np.float32)), axis=1)
            # (imgH - 1 + 1) x imgW x 2
            gradient_y = np.concatenate((np.diff(flow_crop, axis=0), np.zeros((1, flow_crop.shape[1], 2), dtype=np.float32)), axis=0)

            # concatenate gradient_x and gradient_y
            gradient = np.concatenate((gradient_x, gradient_y), axis=2)

            # We can trust the gradient outside of flow_mask_gradient_img
            # We assume the gradient within flow_mask_gradient_img is 0.
            gradient[flow_mask_gradient_img_crop, :] = 0

            # Complete the flow
            imgSrc_gy = gradient[:, :, 2 : 4]
            imgSrc_gy = imgSrc_gy[0 : flow_crop.shape[0] - 1, :, :]
            imgSrc_gx = gradient[:, :, 0 : 2]
            imgSrc_gx = imgSrc_gx[:, 0 : flow_crop.shape[1] - 1, :]
            if mode == 'forward':
                edge_crop=edge[minbbox_tl[i][1]:minbbox_br[i][1],minbbox_tl[i][0]:minbbox_br[i][0],i]
            else:
                edge_crop=edge[minbbox_tl[i+1][1]:minbbox_br[i+1][1],minbbox_tl[i+1][0]:minbbox_br[i+1][0],i]
            compFlow_crop = Poisson_blend(flow_crop, imgSrc_gx, imgSrc_gy, flow_mask_img_crop, edge_crop)

            #return original size
            if mode == 'forward':
                compFlow[minbbox_tl[i][1]:minbbox_br[i][1],minbbox_tl[i][0]:minbbox_br[i][0], :, i] = compFlow_crop
            else:
                compFlow[minbbox_tl[i+1][1]:minbbox_br[i+1][1],minbbox_tl[i+1][0]:minbbox_br[i+1][0], :, i] = compFlow_crop

        else:
            flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
            flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
            compFlow[:, :, :, i] = flow

        ## Flow visualization.
        # flow_img = utils.flow_viz.flow_to_image(compFlow[:, :, :, i])
        # flow_img = Image.fromarray(flow_img)

        ## Saves the flow and flow_img.
        # flow_img.save(os.path.join(args.outroot, 'flow_comp', mode + '_png', '%05d.png'%i))
        # utils.frame_utils.writeFlow(os.path.join(args.outroot, 'flow_comp', mode + '_flo', '%05d.flo'%i), compFlow[:, :, :, i])

    return compFlow


def edge_completion(args, EdgeGenerator, corrFlow, flow_mask, mode):
    """Calculate flow edge and complete it.
    """

    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW, _, nFrame = corrFlow.shape
    Edge = np.empty(((imgH, imgW, 0)), dtype=np.float32)

    for i in range(nFrame):
        print("Completing {0} flow edge {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow_mask_img = flow_mask[:, :, i] if mode == 'forward' else flow_mask[:, :, i + 1]

        flow_img_gray = (corrFlow[:, :, 0, i] ** 2 + corrFlow[:, :, 1, i] ** 2) ** 0.5
        flow_img_gray = flow_img_gray / flow_img_gray.max()

        edge_corr = canny(flow_img_gray, sigma=2, mask=(1 - flow_mask_img).astype(np.bool))
        edge_completed = infer(args, EdgeGenerator, torch.device('cuda:1'), flow_img_gray, edge_corr, flow_mask_img)
        Edge = np.concatenate((Edge, edge_completed[..., None]), axis=-1)

    return Edge


def video_completion_seamless(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # Flow model.
    RAFT_model = initialize_RAFT(args)

    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    # Obtains imgH, imgW and nFrame.
    imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
    nFrame = len(filename_list)

    # Loads video.
    video = []
    for filename in sorted(filename_list):
        video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)).permute(2, 0, 1).float())
    video = torch.stack(video, dim=0)
    video = video.to('cuda')

    # Loads masks.
    filename_list = glob.glob(os.path.join(args.path_mask, '*.png')) + \
                    glob.glob(os.path.join(args.path_mask, '*.jpg'))
    mask = []
    mask_dilated = []
    flow_mask = []
    for filename in sorted(filename_list):
        mask_img = np.array(Image.open(filename).convert('L'))
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=3)
        # Close the small holes inside the foreground objects
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11),np.uint8)).astype(np.bool)
        flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.bool)
        flow_mask.append(flow_mask_img)
        
        # Dilate a little bit 
        mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=3)
        mask_img = scipy.ndimage.binary_fill_holes(mask_img).astype(np.bool)
        mask.append(mask_img)
        mask_dilated.append(gradient_mask(mask_img))

    minbbox_tl,minbbox_br=find_minbbox(flow_mask)

    # Image inpainting model.
    deepfill = DeepFillv1(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW])
    
    # timer
    time_start=time.time()
    
    # Calcutes the corrupted flow.
    corrFlowF = calculate_flow(args, RAFT_model, video, 'forward')
    corrFlowB = calculate_flow(args, RAFT_model, video, 'backward')
    print('\nFinish flow prediction.')

    # Makes sure video is in BGR (opencv) format.
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.

    # mask indicating the missing region in the video.
    mask = np.stack(mask, -1).astype(np.bool)
    mask_dilated = np.stack(mask_dilated, -1).astype(np.bool)
    flow_mask = np.stack(flow_mask, -1).astype(np.bool)
    
    if args.edge_guide:
        # Edge completion model.
        EdgeGenerator = EdgeGenerator_()
        EdgeComp_ckpt = torch.load(args.edge_completion_model)
        EdgeGenerator.load_state_dict(EdgeComp_ckpt['generator'])
        EdgeGenerator.to(torch.device('cuda:1'))
        EdgeGenerator.eval()

        # Edge completion.
        FlowF_edge = edge_completion(args, EdgeGenerator, corrFlowF, flow_mask, 'forward')
        FlowB_edge = edge_completion(args, EdgeGenerator, corrFlowB, flow_mask, 'backward')
        print('\nFinish edge completion.')
    else:
        FlowF_edge, FlowB_edge = None, None

   
    # Completes the flow.
    videoFlowF = complete_flow(args, corrFlowF, flow_mask, 'forward', minbbox_tl, minbbox_br, FlowF_edge)
    videoFlowB = complete_flow(args, corrFlowB, flow_mask, 'backward', minbbox_tl, minbbox_br, FlowB_edge)
    print('\nFinish flow completion.')

    # Prepare gradients
    gradient_x = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    gradient_y = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)

    for indFrame in range(nFrame):
        img = video[:, :, :, indFrame]
        img[mask[:, :, indFrame], :] = 0         

        img = cv2.inpaint((img * 255).astype(np.uint8), mask[:, :, indFrame].astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)  / 255.
        
        gradient_x_ = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
        gradient_y_ = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
        gradient_x = np.concatenate((gradient_x, gradient_x_.reshape(imgH, imgW, 3, 1)), axis=-1)
        gradient_y = np.concatenate((gradient_y, gradient_y_.reshape(imgH, imgW, 3, 1)), axis=-1)

        gradient_x[mask_dilated[:, :, indFrame], :, indFrame] = 0
        gradient_y[mask_dilated[:, :, indFrame], :, indFrame] = 0


    iter = 0
    mask_tofill = mask
    gradient_x_filled = gradient_x # corrupted gradient_x, mask_gradient indicates the missing gradient region
    gradient_y_filled = gradient_y # corrupted gradient_y, mask_gradient indicates the missing gradient region
    mask_gradient = mask_dilated
    video_comp = video

    # We iteratively complete the video.
    while(np.sum(mask) > 0):
     #   create_dir(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter)))

        # Gradient propagation.
        gradient_x_filled, gradient_y_filled, mask_gradient = \
            get_flowNN_gradient(args,
                                gradient_x_filled,
                                gradient_y_filled,
                                mask,
                                mask_gradient,
                                videoFlowF,
                                videoFlowB,
                                None,
                                None)

        # if there exist holes in mask, Poisson blending will fail. So I did this trick. I sacrifice some value. Another solution is to modify Poisson blending.
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = scipy.ndimage.binary_fill_holes(mask_gradient[:, :, indFrame]).astype(np.bool)

        # After one gradient propagation iteration
        # gradient --> RGB
        for indFrame in range(nFrame):
            print("Poisson blending frame {0:3d}".format(indFrame))

            if mask[:, :, indFrame].sum() > 0:
                try:
                    video_comp_crop=video_comp[minbbox_tl[indFrame][1]:minbbox_br[indFrame][1],minbbox_tl[indFrame][0]:minbbox_br[indFrame][0], :, indFrame]
                    gradient_x_filled_crop=gradient_x_filled[minbbox_tl[indFrame][1]:minbbox_br[indFrame][1],minbbox_tl[indFrame][0]:minbbox_br[indFrame][0]-1, :, indFrame]
                    gradient_y_filled_crop=gradient_y_filled[minbbox_tl[indFrame][1]:minbbox_br[indFrame][1]-1,minbbox_tl[indFrame][0]:minbbox_br[indFrame][0], :, indFrame]
                    mask_crop=mask[minbbox_tl[indFrame][1]:minbbox_br[indFrame][1],minbbox_tl[indFrame][0]:minbbox_br[indFrame][0], indFrame]
                    mask_gradient_crop=mask_gradient[minbbox_tl[indFrame][1]:minbbox_br[indFrame][1],minbbox_tl[indFrame][0]:minbbox_br[indFrame][0], indFrame];
                    frameBlend_crop, UnfilledMask_crop = Poisson_blend_img(video_comp_crop, gradient_x_filled_crop, gradient_y_filled_crop, mask_crop, mask_gradient_crop)

                    frameBlend, UnfilledMask = video_comp[:, :, :, indFrame], mask[:, :, indFrame]
                    frameBlend[minbbox_tl[indFrame][1]:minbbox_br[indFrame][1],minbbox_tl[indFrame][0]:minbbox_br[indFrame][0],:]=frameBlend_crop
                    UnfilledMask[minbbox_tl[indFrame][1]:minbbox_br[indFrame][1],minbbox_tl[indFrame][0]:minbbox_br[indFrame][0]]=UnfilledMask_crop
                    # UnfilledMask = scipy.ndimage.binary_fill_holes(UnfilledMask).astype(np.bool)
                    
                    # frameBlend, UnfilledMask = Poisson_blend_img(video_comp[:, :, :, indFrame], gradient_x_filled[:, 0 : imgW - 1, :, indFrame], gradient_y_filled[0 : imgH - 1, :, :, indFrame], mask[:, :, indFrame], mask_gradient[:, :, indFrame])
                  #  UnfilledMask = scipy.ndimage.binary_fill_holes(UnfilledMask).astype(np.bool)
                except:
                    frameBlend, UnfilledMask = video_comp[:, :, :, indFrame], mask[:, :, indFrame]

                frameBlend = np.clip(frameBlend, 0, 1.0)
                tmp = cv2.inpaint((frameBlend * 255).astype(np.uint8), UnfilledMask.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.
                frameBlend[UnfilledMask, :] = tmp[UnfilledMask, :]

                video_comp[:, :, :, indFrame] = frameBlend
                mask[:, :, indFrame] = UnfilledMask

                frameBlend_ = copy.deepcopy(frameBlend)
                # Green indicates the regions that are not filled yet.
                frameBlend_[mask[:, :, indFrame], :] = [0, 1., 0]
            else:
                frameBlend_ = video_comp[:, :, :, indFrame]

        # cv2.imwrite(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter), '%05d.png'%indFrame), frameBlend_ * 255.)

        # video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
        # imageio.mimwrite(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter), 'intermediate_{0}.mp4'.format(str(iter))), video_comp_, fps=12, quality=8, macro_block_size=1)
        # imageio.mimsave(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter), 'intermediate_{0}.gif'.format(str(iter))), video_comp_, format='gif', fps=12)

        mask, video_comp = spatial_inpaint(deepfill, mask, video_comp)
        iter += 1

        # Re-calculate gradient_x/y_filled and mask_gradient
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = gradient_mask(mask[:, :, indFrame])

            gradient_x_filled[:, :, :, indFrame] = np.concatenate((np.diff(video_comp[:, :, :, indFrame], axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
            gradient_y_filled[:, :, :, indFrame] = np.concatenate((np.diff(video_comp[:, :, :, indFrame], axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)

            gradient_x_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0
            gradient_y_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0

    video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
 
    time_end=time.time()
    print('time cost',time_end-time_start,'s')

    # write out
    create_dir(os.path.join(args.outroot, 'frame_seamless_comp_' + 'final'))
    for i in range(nFrame):
        img = video_comp[:, :, :, i] * 255
        cv2.imwrite(os.path.join(args.outroot, 'frame_seamless_comp_' + 'final', '%06d.png'%i), img)
        # imageio.mimwrite(os.path.join(args.outroot, 'frame_seamless_comp_' + 'final', 'final.mp4'), video_comp_, fps=12, quality=8, macro_block_size=1)
        # imageio.mimsave(os.path.join(args.outroot, 'frame_seamless_comp_' + 'final', 'final.gif'), video_comp_, format='gif', fps=12)


def main(args):

    assert args.mode in ('object_removal', 'video_extrapolation'), (
        "Accepted modes: 'object_removal', 'video_extrapolation', but input is %s"
    ) % mode

    if args.seamless:
        video_completion_seamless(args)
    else:
        video_completion(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video completion
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='../data/frames_corr', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='../data/masks', help="mask for object removal")
    parser.add_argument('--outroot', default='../result/', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)

    # RAFT
    parser.add_argument('--model', default='./weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Deepfill
    parser.add_argument('--deepfill_model', default='./weight/imagenet_deepfill.pth', help="restore checkpoint")

    # Edge completion
    parser.add_argument('--edge_completion_model', default='./weight/edge_completion.pth', help="restore checkpoint")

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    args = parser.parse_args()    

    main(args)
    
