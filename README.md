# Baseline
本代码的基本流程：计算稠密光流（RAFT）-> 计算边缘（Canny）-> 补全边缘（EdgeConnect）-> 补全光流（解Ax=b）-> 传播RGB值(能从时序上拿来补的像素就拿来补；没有的话就拿最空的一帧来进行图像补全(Deepfill_V1)后再传播) <br/><br/>
在基于原作代码的基础上，为了增速，我修改了一下其中的光流补全部分。原作是全图去解Ax=b，特别特别慢。我改成了crop后再进去，解Ax=b会快些。<br/>

用法：
```bash
python ./tool/video_completion_modified.py --mode object_removal --path ../data/test_a/video_0000/frames_corr --path_mask ../data/test_a/video_0000/masks --outroot ../data/result_test_a/video_0000 --seamless --edge_guide
```
<br/>

该baseline：<br/>
1) 在test_a数据集上，本baseline的最终分数约为68.7054。<br/>
2) 速度慢。CPU:Gold5218@2.30GHz, GPU:NVIDIA-V100, 无多线程跑了一天多。<br/>
3) 效果不均。在部分视频上效果较好(e.g.舞动的人，移动的物体等，因有时序上的信息可补足)，在部分视频上效果较差(e.g.水印，固定位置的物体等)。<br/>

温馨提示：<br/>
1) 在比赛官网提交结果时，顶上将有进度条，且提交成功后会有提示。接收到"提交成功"的提示前不要关掉页面哦。<br/>
2) 评分失败时提示"folder number unmatch"的错误时，原因可能有以下两个: 1.即为提交的视频文件夹数量有错, video_*** 的数量要100个。请查看是否多了不相关文件夹/或者是少了某些视频文件夹; 2. 请直接从内部打包，即result.zip解压后即为 result/video_**** 而不是 aaa/result/video_**** <br/>
3) 评分失败时提示"image not found"的错误时，请检查每个文件夹里的图片个数是否完整，命名是否正确。 <br/>
4) 评分失败时提示"image size unmatch"的错误时，请检查图片大小是否如bbox.txt所示。 <br/>
<br/><br/>

# 第二届“马栏山杯”国际音视频算法大赛-视频补全介绍

## 数据说明
**训练集**：2194个视频，格式为mp4，视频时长2s~8s。视频大小主要为720x1280和1080x1920。<br/>
train1（843个视频，压缩包大小1.98GB），与去年点位跟踪赛道research_1视频集相同（去年数据集里的重复视频现已去重）。<br/>
train2（1010个视频，压缩包大小2.18GB），与去年点位跟踪赛道research_2视频集相同（去年数据集里的重复视频现已去重）。<br/>
train3（341个视频，压缩包大小646M），为去年点位跟踪赛道val视频集 + 本赛道新视频。<br/>

**验证集**：50个视频，格式为png图像，图像大小为576x1024（我们尽量裁剪掉了角标和字幕区域），帧数120帧及以下。每个样本由原视频和任意一种mask（整块挖空、局部水印、随机噪声块、人像）组成。

**测试集a**： 100个挖空视频，格式为png图像，图像大小为576x1024（我们尽量裁剪掉了角标和字幕区域），帧数120帧及以下。每个样本包含成对的挖空原视频和其相对应的mask。<br/>
**测试集b**： 格式同测试集a。<br/>
（测试集的各个mask类型所占的比例与验证集相似。）

本竞赛要求整个竞赛过程中不能采用第三方数据，可以使用开源的预训练模型（且star数超过500）。<br/>
验证集允许加入训练，测试集**禁止**加入训练。<br/>
本大赛提供的数据版权归芒果TV所有，参赛选手不能将其泄露或用于本大赛外其他用途。<br/>

## 评估指标
初赛和复赛通过评估选手提交的结果来评分，本次比赛采用PSNR和SSIM两种评价指标。对于上传的结果，评估程序将计算挖空区域的PSNR和SSIM两种指标，均采用逐帧计算并进行平均。最终，PSNR和SSIM进行加权计算，并得到最终竞赛得分。PSNR取值在[0, 80]，优秀分值范围大约在[30, 50]；SSIM取值在[0, 1]，优秀分值范围大约在[0.8, 1]
```bash
score = PSNR*2*0.5 + SSIM*100*0.5
```
PSNR和SSIM的具体计算方式可看evaluate.py

## 算力要求
1) 前向推理时的内存使用不超过8G，显存使用不超过8G。
2) 在CPU（双核，2.30GHz），GPU（单卡，NVIDIA V100或NVIDIA 3090），按顺序跑完测试集（100个视频，不开多线程）的时间不超过15小时。
<br/>
未满足以上限制的参赛队伍，大赛官方有权将最终总成绩认定为无效，排名由后一名依次递补。

## 作品提交要求
初赛和复赛的结果提交方式相同，都需要提交裁剪后的图片。为了缩小上传结果的大小，选手需根据各个视频内提供的bbox.txt提供的裁剪框[x,y,w,h]裁剪出对应的结果图片并打包上传。
其中，x,y的索引从0开始。例如：
```bash
import cv2
img = cv2.imread("result_000000.png")
crop_img = img[y:y+h, x:x+w, :]
cv2.imwrite("crop_000000.png", crop_img)
```
选手需要将裁剪后的图片文件按放入各个视频文件夹（video_0***），最后一起打包成*.zip格式后上传（正常大小不超过2G）。请直接在内部打包，即result.zip解压后即为 result/video_0*** 而不是 aaa/result/video_0***。<br/>
文件夹结构和命名规则如下(以test_a为例) ("result"可任意命名, 你可以随意命名成result_0608, aaa_123等等) ：
```bash
                      |—— crop_000000.png
       |—— video_0000 |—— crop_000001.png
       |              |—— crop_...
result |—— video_...
       |              |—— crop_000000.png
       |—— video_0099 |—— crop_000001.png
                      |—— crop_...
```
## 评测及排行
1) 初赛和复赛均提供下载数据，选手在本地进行算法调试，在比赛页面提交结果；
2) 初赛和复赛采用AB榜机制，A榜成绩供参赛队伍在比赛中查看，最终比赛排名（包括初赛和复赛）采用B榜最佳成绩；
3) 复赛TOP10团队需提供源代码、以及docker镜像，供大赛组委会进行方案和结果验证，只有复现结果的性能指标和提交最优结果的性能指标差异在允许的范围内，选手的成绩才是真实有效的；
4) 复赛TOP10团队必须提交一份**技术方案报告**来阐述详细方案 (建议长度1-4页)
5) 每支团队每天最多提交3次；
6) 排行按照得分从高到低排序，排行榜将选择团队的历史最优成绩进行排名；

<br/><br/><br/>
下面是原作的README.md
# [ECCV 2020] Flow-edge Guided Video Completion

### [[Paper](https://arxiv.org/abs/2009.01835)] [[Project Website](http://chengao.vision/FGVC/)] [[Google Colab](https://colab.research.google.com/drive/1pb6FjWdwq_q445rG2NP0dubw7LKNUkqc?usp=sharing)]

<p align='center'>
<img src='http://chengao.vision/FGVC/files/FGVC_teaser.png' width='900'/>
</p>

We present a new flow-based video completion algorithm. Previous flow completion methods are often unable to retain the sharpness of motion boundaries. Our method first extracts and completes motion edges, and then uses them to guide piecewise-smooth flow completion with sharp edges. Existing methods propagate colors among local flow connections between adjacent frames. However, not all missing regions in a video can be reached in this way because the motion boundaries form impenetrable barriers. Our method alleviates this problem by introducing non-local flow connections to temporally distant frames, enabling propagating video content over motion boundaries. We validate our approach on the DAVIS dataset. Both visual and quantitative results show that our method compares favorably against the state-of-the-art algorithms.
<br/>

**[ECCV 2020] Flow-edge Guided Video Completion**
<br/>
[Chen Gao](http://chengao.vision), [Ayush Saraf](#), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), and [Johannes Kopf](https://johanneskopf.de/)
<br/>
In European Conference on Computer Vision (ECCV), 2020

## Prerequisites

- Linux (tested on CentOS Linux release 7.4.1708)
- Anaconda
- Python 3.8 (tested on 3.8.5)
- PyTorch 1.6.0

and the Python dependencies listed in requirements.txt

- To get started, please run the following commands:
  ```
  conda create -n FGVC
  conda activate FGVC
  conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch
  conda install matplotlib scipy
  pip install -r requirements.txt
  ```

- Next, please download the model weight and demo data using the following command:
  ```
  chmod +x download_data_weights.sh
  ./download_data_weights.sh
  ```

## Quick start

- Object removal:
```bash
cd tool
python video_completion.py \
       --mode object_removal \
       --path ../data/tennis \
       --path_mask ../data/tennis_mask \
       --outroot ../result/tennis_removal \
       --seamless
```

- FOV extrapolation:
```bash
cd tool
python video_completion.py \
       --mode video_extrapolation \
       --path ../data/tennis \
       --outroot ../result/tennis_extrapolation \
       --H_scale 2 \
       --W_scale 2 \
       --seamless
```

You can remove the `--seamless` flag for a faster processing time.


## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

If you find this code useful for your research, please consider citing the following paper:

	@inproceedings{Gao-ECCV-FGVC,
	    author    = {Gao, Chen and Saraf, Ayush and Huang, Jia-Bin and Kopf, Johannes},
	    title     = {Flow-edge Guided Video Completion},
	    booktitle = {European Conference on Computer Vision},
	    year      = {2020}
	}

## Acknowledgments
- Our flow edge completion network builds upon [EdgeConnect](https://github.com/knazeri/edge-connect).
- Our image inpainting network is modified from [DFVI](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting).
