# Adaptive Transformers for Robust Few-shot Cross-domain Face Anti-spoofing
Contact: Hsin-Ping Huang (hhuang79@ucmerced.edu)

## Introduction

While recent face anti-spoofing methods perform well under the intra-domain setups, an effective approach needs to account for much larger appearance variations of images acquired in complex scenes with different sensors for robust performance. In this paper, we present adaptive vision transformers (ViT) for robust cross-domain face anti-spoofing. Specifically, we adopt ViT as a backbone to exploit its strength to account for long-range dependencies among pixels. We further introduce the ensemble adapters module and feature-wise transformation layers in the ViT to adapt to different domains for robust performance with a few samples. Experiments on several benchmark datasets show that the proposed models achieve both robust and competitive performance against the state-of-the-art methods.

## Paper

[Adaptive Transformers for Robust Few-shot Cross-domain Face Anti-spoofing](https://arxiv.org/abs/2203.12175) <br />
[Hsin-Ping Huang](https://hhsinping.github.io/), [Deqing Sun](https://deqings.github.io/), [Yaojie Liu](https://yaojieliu.github.io/), [Wen-Sheng Chu](https://l2ior.github.io/), [Taihong Xiao](https://prinsphield.github.io/), [Jinwei Yuan](https://www.linkedin.com/in/jinwei-yuan-9777185b/), [Hartwig Adam](https://research.google/people/author37870/), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html) <br />


European Conference on Computer Vision (ECCV), 2022 <br />

Please cite our paper if you find it useful for your research.

```
@inproceedings{huang_2022_adaptive,
   title = {Adaptive Transformers for Robust Few-shot Cross-domain Face Anti-spoofing},
   author={Huang, Hsin-Ping and Sun, Deqing and Liu, Yaojie, and Chu, Wen-Sheng and Xiao, Taihong and Yuan, Jinwei and Adam, Hartwig and Yang, Ming-Hsuan},
   booktitle = {ECCV},
   year={2022}
}
```
## Installation and Usage
### Clone this repo.
```
git clone https://github.com/hhsinping/few_shot_fas.git
cd few_shot_fas
```
### Install the packages.
- Create conda environment and install required packages.
1. Python 3.7
2. Pytorch 1.7.1, Torchvision 0.8.2, timm 0.4.9
3. Pandas, Matplotlib, Opencv, Sklearn
```
conda create -n fas python=3.7.4 -y
conda activate fas
pip install torch==1.7.1 torchvision==0.8.2 timm==0.4.9
pip install pandas scikit-learn matplotlib opencv-python
```
- Clone the external repo timm and copy our code there
```
git clone https://github.com/rwightman/pytorch-image-models.git
cd pytorch-image-models && git checkout e7f0db8 && cd ../
mv pytorch-image-models/timm ./third_party && rm pytorch-image-models -r && mv vision_transformer.py third_party/models && mv helpers.py third_party/models
```
- Download and extract the file lists of external data for protocol 1 and protocol 2
```
wget https://www.dropbox.com/s/2mxh5r8hf0m8m1n/data.tgz
tar zxvf data.tgz
```

### Datasets
- To run the code, you will need to request the following datasets.

1. Protocol 1: [MSU-MFSD](https://sites.google.com/site/huhanhomepage/download/), [CASIA-MFSD](https://ieeexplore.ieee.org/document/6199754), [Replay-attack](https://www.idiap.ch/en/dataset/replayattack), [CelebA-Spoof](https://drive.google.com/corp/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z), [OULU-NPU](https://sites.google.com/site/oulunpudatabase/)
2. Protocol 2: [WMCA](https://www.idiap.ch/en/dataset/wmca), [CASIA-SURF CeFA](https://sites.google.com/corp/qq.com/face-anti-spoofing/dataset-download/casia-surf-cefacvpr2020), [CASIA-SURF](https://sites.google.com/corp/qq.com/face-anti-spoofing/dataset-download/casia-surfcvpr2019)

- For protocol 1, we follow the preprocessing step of [SSDG](https://github.com/taylover-pei/SSDG-CVPR2020) which detect and align the faces using [MTCNN](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection). 
1. For each video, we only sample two frames: frame[6] and frame[6+math.floor(total_frames/2)] and save the frame as videoname_frame0.png/videoname_frame1.png, except for the CelebA-Spoof dataset.
2. We input the sample frames into MTCNN to detect, align and crop the images. The image are resized to (224,224,3), only the RGB channels are used.
3. We save the frames into data/MCIO/frame/ following the file name listed in data/MCIO/txt/, we provide the file structure below:
   ```
   data/MCIO/frame/
   |-- casia
       |-- train
       |   |--real
       |   |  |--1_1_frame0.png, 1_1_frame1.png 
       |   |--fake
       |      |--1_3_frame0.png, 1_3_frame1.png 
       |-- test
           |--real
           |  |--1_1_frame0.png, 1_1_frame1.png 
           |--fake
              |--1_3_frame0.png, 1_3_frame1.png 
   |-- msu
       |-- train
       |   |--real
       |   |  |--real_client002_android_SD_scene01_frame0.png, real_client002_android_SD_scene01_frame1.png
       |   |--fake
       |      |--attack_client002_android_SD_ipad_video_scene01_frame0.png, attack_client002_android_SD_ipad_video_scene01_frame1.png
       |-- test
           |--real
           |  |--real_client001_android_SD_scene01_frame0.png, real_client001_android_SD_scene01_frame1.png
           |--fake
              |--attack_client001_android_SD_ipad_video_scene01_frame0.png, attack_client001_android_SD_ipad_video_scene01_frame1.png
   |-- replay
       |-- train
       |   |--real
       |   |  |--real_client001_session01_webcam_authenticate_adverse_1_frame0.png, real_client001_session01_webcam_authenticate_adverse_1_frame1.png
       |   |--fake
       |      |--fixed_attack_highdef_client001_session01_highdef_photo_adverse_frame0.png, fixed_attack_highdef_client001_session01_highdef_photo_adverse_frame1.png
       |-- test
           |--real
           |  |--real_client009_session01_webcam_authenticate_adverse_1_frame0.png, real_client009_session01_webcam_authenticate_adverse_1_frame1.png
           |--fake
              |--fixed_attack_highdef_client009_session01_highdef_photo_adverse_frame0.png, fixed_attack_highdef_client009_session01_highdef_photo_adverse_frame1.png
   |-- oulu
       |-- train
       |   |--real
       |   |  |--1_1_01_1_frame0.png, 1_1_01_1_frame1.png
       |   |--fake
       |      |--1_1_01_2_frame0.png, 1_1_01_2_frame1.png
       |-- test
           |--real
           |  |--1_1_36_1_frame0.png, 1_1_36_1_frame1.png
           |--fake
              |--1_1_36_2_frame0.png, 1_1_36_2_frame1.png
   |-- celeb
       |-- real
       |   |--167_live_096546.jpg
       |-- fake
           |--197_spoof_420156.jpg       
   ```

- For protocol 2, we use the original frames and cut the black borders as inputs.

1. We use all the frames in surf dataset with their original file names. We sample 10 frames in each video equidistantly for cefa and wmca datasets. We save the sampled frame as videoname_XX.jpg (where XX denotes the index of sampled frame, detailed file names can be found in data/WCS/txt/.
2. We [cut the black borders](https://github.com/AlexanderParkin/CASIA-SURF_CeFA/blob/205d3d976523ed0c15d1e709ed7f21d50d7cf19b/at_learner_core/at_learner_core/utils/transforms.py#L456) of the images and input to our code. The images are then resized to (224,224,3), only the RGB channels are used.
3. We save the frames into data/WCS/frame/ following the file name listed in data/WCS/txt/, we provide the file structure below:
   ```
   data/WCS/frame/
   |-- wmca
       |-- train
       |   |--real
       |   |  |--31.01.18_035_01_000_0_01_00.jpg, 31.01.18_035_01_000_0_01_05.jpg
       |   |--fake
       |      |--31.01.18_514_01_035_1_05_00.jpg, 31.01.18_514_01_035_1_05_05.jpg
       |-- test
           |--real
           |  |--31.01.18_036_01_000_0_00_00.jpg, 31.01.18_036_01_000_0_00_01.jpg
           |--fake
              |--31.01.18_098_01_035_3_13_00.jpg, 31.01.18_098_01_035_3_13_01.jpg
   |-- cefa
       |-- train
       |   |--real
       |   |  |--3_499_1_1_1_00.jpg, 3_499_1_1_1_01.jpg
       |   |--fake
       |      |--3_499_3_2_2_00.jpg, 3_499_3_2_2_01.jpg
       |-- test
           |--real
           |  |--3_299_1_1_1_00.jpg, 3_299_1_1_1_01.jpg
           |--fake
              |--3_299_3_2_2_00.jpg, 3_299_3_2_2_01.jpg
   |-- surf
       |-- train
       |   |--real
       |   |  |--Training_real_part_CLKJ_CS0110_real.rssdk_color_91.jpg
       |   |--fake
       |      |--Training_fake_part_CLKJ_CS0110_06_enm_b.rssdk_color_91.jpg
       |-- test
           |--real
           |  |--Val_0007_007243-color.jpg
           |--fake
              |--Val_0007_007193-color.jpg
   ```


## Training example
To train the network you can run
```
python train.py --config [C/M/I/O/cefa/surf/wmca]
```
Saved model can be found as casia_checkpoint.pth.tar, msu_checkpoint.pth.tar, replay_checkpoint.pth.tar, oulu_checkpoint.pth.tar, cefa_checkpoint.pth.tar, surf_checkpoint.pth.tar, wmca_checkpoint.pth.tar.

## Testing example
To evaluate the network you can run
```
python test.py --config [C/M/I/O/cefa/surf/wmca]
```
The test script will read the checkpoint msu_checkpoint.pth.tar, replay_checkpoint.pth.tar, oulu_checkpoint.pth.tar, cefa_checkpoint.pth.tar, surf_checkpoint.pth.tar, wmca_checkpoint.pth.tar, and the test AUC/HTER/TPR@FPR will be printed.

## Config
The file config.py contains the hyper-parameters used during training/testing.

## Acknowledgement
The implementation is partly based on the following projects: [SSDG](https://github.com/taylover-pei/SSDG-CVPR2020), [timm](https://github.com/rwightman/pytorch-image-models), [BERT on STILTs](https://github.com/zphang/bert_on_stilts), [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot).

