'''
线上测试使用的是GPU版本(调的是piq库)。这里面我们也提供了CPU版的测试代码(调的是skimage库)。
CPU版本和GPU版本分值有差别：此baseline的cpu测评结果为68.6891, gpu测评结果为68.7054

error_code:  
-1 error: video number unmatch
-2 error: image not found
-3 error: image size unmatch
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import argparse
import glob
from PIL import Image
import torch
import cv2
from piq import ssim, SSIMLoss
from piq import psnr

# CPU版本
def PSNR(ximg,yimg):
    return compare_psnr(ximg,yimg,data_range=255)

def SSIM(y,t,value_range=255):   
    try:
        ssim_value = ssim(y, t, gaussian_weights=True, data_range=value_range, multichannel=True)
    except ValueError:
        #WinSize too small
        ssim_value = ssim(y, t, gaussian_weights=True, data_range=value_range, multichannel=True, win_size=3)
    return ssim_value

# GPU版本。
# def PSNR(ximg,yimg):
#     gt_tensor = torch.from_numpy(yimg)
#     img_tensor = torch.from_numpy(ximg)
#     psnr_v = psnr(img_tensor.cuda(), gt_tensor.cuda(),data_range=255.).item()
#     #print(psnr_v)
#     return psnr_v

# def SSIM(y,t,value_range=255):   
#     gt_ss_tensor = torch.from_numpy(y.transpose([2,0,1])).unsqueeze(0).cuda()
#     img_ss_tensor = torch.from_numpy(t.transpose([2,0,1])).unsqueeze(0).cuda()
#     loss = SSIMLoss(data_range=255.).cuda()
#     loss_v = loss(gt_ss_tensor, img_ss_tensor).item()
#     return 1-loss_v


def Evaluate(files_gt, files_pred, methods = [PSNR,SSIM]):
    score = {}
    for meth in methods:
        name = meth.__name__
        results = []
        res=0
        frame_num=len(files_gt)
        for frame in range(0,frame_num):
            # ignore some tiny crops 
            if files_gt[frame].shape[0]*files_gt[frame].shape[1]<150:    
                continue

            res = meth(files_pred[frame],files_gt[frame])
            results.append(res)        

        mres = np.mean(results)
        stdres=np.std(results)
        # print(name+": "+str(mres)+" Std: "+str(stdres))
        score['mean']=mres
        score['std']=stdres
    return score


def evaluate(args):   
    error_code=0
    error_flag='successful.'
    final_result=[]
    
    # load video folder
    grountruth_folder_list = sorted(glob.glob(os.path.join(args.groundtruth_folder, 'video_0*')))
    prediction_folder_list = sorted(glob.glob(os.path.join(args.prediction_folder,'video_0*')))    
    
    if len(grountruth_folder_list) != len(prediction_folder_list): 
        error_code=-1
        error_flag='folder number unmatch.'
        return error_code, error_flag, 0    
    
    for i in range(0,len(grountruth_folder_list)):
        # load video
        video_gt=[]
        video_predict=[]
        image_list = sorted(glob.glob(os.path.join(grountruth_folder_list[i],'gt_crop/*.png')))
        for image_gt in image_list:
            video_gt.append(np.array(Image.open(image_gt)).astype(np.uint8))
               
            try: 
                image_predict=prediction_folder_list[i]+'/crop_'+image_gt[-10:]
                video_predict.append(np.array(Image.open(image_predict)).astype(np.uint8))
            except:
                error_code=-2
                error_flag= 'read ' + image_predict +' failed.'
                return error_code, error_flag, 0
        
        # check image size
        for j in range(0,len(image_list)):
            if video_gt[j].shape!=video_predict[j].shape:
                error_code=-3
                error_flag= 'image size unmatch. please check video_' + str(i).zfill(4)+'/crop_'+str(j).zfill(4)
                return error_code, error_flag, 0

        # sent in whole video
        psnr_res = Evaluate(video_gt,video_predict, methods=[PSNR])
        ssim_res = Evaluate(video_gt,video_predict, methods=[SSIM])

        psnr_res_norm=min(80,psnr_res['mean'])           
        ssim_res_norm=ssim_res['mean']*100

        result=psnr_res_norm+ssim_res_norm*0.5
        print(i,psnr_res_norm,ssim_res_norm,result)

        final_result.append(result)
    return error_code, error_flag, np.mean(final_result) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groundtruth_folder',default='./')
    parser.add_argument('--prediction_folder',default='./')
    # usage: python scoring.py --groundtruth_folder ./val --prediction_folder ./result   

    # groundtruth文件夹结构如下(跟解压后的一样)：
    #     
    #                                    |-- 000000.png
    #       |-- video_0000 -- gt_crops --|-- 000001.png
    #       |                            |-- ...
    #  val -|-- video_0001  
    #       |
    #       |-- ...

    # 选手们上传的文件夹结构如下：
    #
    #                          |-- crop_000000.png
    #         |-- video_0000 --|-- crop_000001.png
    #         |                |-- ...
    # result -|-- video_0001  
    #         |
    #         |-- ...

    args = parser.parse_args()
    error_code, error_flag, final_result = evaluate(args)
    print(final_result)


