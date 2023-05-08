# 2023 [DFGC-VRA](https://codalab.lisn.upsaclay.fr/competitions/10754) Solution 
This repo provides an solution for the DeepFake Game Competition on Visual Realism Assessment (DFGC-VRA) @ IJCB 2023. Our solution achieve the 7st in the final phase of the DFGC-VRA. The ranking can be seen [here](https://codalab.lisn.upsaclay.fr/competitions/10754#results).
## 1. Authors

Institution: China Nanhu Academy of Electronics and Information Technology(CNAEIT)

Adviser: Zhifeng Xiao; Shu Xu

Username: xuenhui

Team members:

- Enhui Xu
- Jincai Xu

##  2. A brief report
Main feature extractor reference [DFGC-2022-1st-place](https://github.com/chenhanch/DFGC-2022-1st-place):
- **Model structure**：ConvNext（convnext_xlarge_384_in22ft1k）and SwinTransformer(swin_large_patch4_window12_384_in22k), with weights pretrained on ImageNet dataset。

- **Ensemble methods**：Two ConvNext at different epochs and one Swin-Transformer.

- **Augmentation methods**：HorizontalFlip、GaussNoise、GaussianBlur

- **Data processing**：A face detector MTCNN is used to crop the face images from video frame (enlarged the face region by a factor of 1.3). Resize the input shape to (3,384,384).

- **Training losses**：BCELoss

Other feature extraction:
- **File naming features**:Segment 5 features for machine learning
- **The intercepted facial features**:(256*256) into a one-dimensional vector
- **Training model**:XGBRegressor
- **Training strategy**: Grid Search
- **Ensemble strategy**: 
  - Feature splicing: fea340+name_fea
  - Result ensemble: Averaging the results of XGB and SVR

References and open-source resources.

  [1] FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV 2019

  [2] https://johann.wang/HifiFace/

  [3] In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking. WIFS 2018.

  [4] Vulnerability Assessment and Detection of Deepfake Videos. ICB 2019

  [5] DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection

  [6] https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html

  [7] Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics. CVPR2020.

  [8] WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection. 2020 ACM MM.

  [9] The DeepFake Detection Challenge (DFDC) Dataset. Arxiv 2020.
  
  [10] https://github.com/chenhanch/DFGC-2022-1st-place
  
## 3. Usage
  
### 3.1 Video frame extraction

- Extract a picture every 5 frames, and save the frame extraction results in the `data_picture` folder

```python
Video_frame_extraction.ipynb
```
### 3.2 Extract faces from `data_picture`

- Use face-recognition to detect face from one folder's different folders' images & save them in the `data_face` folder

```python
findfaceFR_folder.py
```
### 3.3 Train and predict the score of test1
- This solution uses only machine learning methods to "blind guess" from file name information. Specifically, the name of the video file is extracted and divided into segments, and a total of 5 features are extracted, and after feature encoding, they are sent to XGB for training. Use grid search to get the highest score of test1（0.7501, 0.7179）.

```python
test1_name_fea_xgb_grid.ipynb
```

### 3.4 Train and predict the score of test2

- The scheme uses the pre-processed face data in the `data_face` folder, each image has a size of 255*255, without resize, and directly trains it as a 196608-dimensional feature vector, using xgb with the original parameters to obtain the best test2 score (0.8610, 0.8850).

```python
test2_pic_xgb_mean.ipynb
```

### 3.5 Train and predict the score of test3
- The 340-dimensional feature vector was selected using the baseline scheme and combined with the 5 name features extracted from the test1 scheme to form a final 345-dimensional vector, which was modeled using XGB, and the results after grid search were fused with those of the baseline method (using fea340). The highest scoring test3 was obtained (0.5911, 0.6656).

```python
train_and_pred.py
test3_fea340+namefea_XGBr.ipynb
```
  
  
## 4. Environment

```python
torch==1.9.0
```
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  


