# 2D Hand Pose Estimation based on CLAMP model

**CLAMP** is original CLAMP project, and **CLAMP_hand** is our project based on CLAMP proejct.

Utilizing the CLAMP model architecture, we achieved single-hand keypoint detection with 0.442 AP(average precision) and dual-hand keypoint detection with 0.53 AP(average precision). We have shown that **accurate hand keypoints can be obtained based on images and prompts**.

<img width="350" alt="image" src="https://github.com/Hand-CLAMP/hand-clamp/assets/94193480/4c6fa57c-f253-4548-b207-ff10d4d0fbda">

## **Motivation** **&** **Problem statement**

- Research on **hand keypoints is gaining importance** as it can be applied to various real-life scenarios such as VR games, robot hands, sign language interpretation, and text generation from sign language videos.
- Commonly used techniques involve estimating hand gesture information through the extraction of hand skeleton features from visual data. However, these methods have limitations in accurately recognizing complex movements and can only identify specific or pre-determined hand shapes.
- We aim to leverage the **CLAMP model** to **obtain precise hand keypoint data** based on **images and prompts**.
- This approach seeks to enhance the accuracy of hand keypoints beyond the recognition of predetermined hand shapes, allowing for more versatile applications in different domains.

**CLAMP Model**

- CLAMP model is used for **animal pose estimation** and its model **used pre-trained language model named CLIP** and it provides rich prior knowledge for describing animal keypoints in text. For example, if there are texts named nose, eyes, mouth, legs, it helps to estimate animal keypoints. Animal keypoints texts will be changed to hand keypoints texts.

**CLIP model**

- This model, developed by OpenAI, is an unsupervised learning model that focuses on **learning the correlation between images and text**.
- Pre-trained on large amounts of image and text data, the model learns the relationships between images and text, making it applicable to various natural language processing and computer vision tasks.

**The reason we use CLAMP model**

- CLAMP model is one of the **most proficient models** in identifying keypoints of animals.
- Since the CLAMP model predicts keypoints of images based on a text prompt, We thought that by **changing the prompt to one related to hands**, it would be possible to capture the keypoints of hand images.

## File Setting & Download

```bash
CLMAP_hand
├─ data
│		└─ hands
│				├─ annotations
│				└─ data
├─ work_dirs
│		├─ CLAMP_ViTB_ap10k_256x256_onehand
│		├─ CLAMP_ViTB_ap10k_256x256
│		└─ CLAMP_ViTB_ap10k_256x256_twohands
│
└─ pretrained
```

You can download the files that need to be placed corresponding folder here: 

- CLAMP_hand/data/hands/annotations : https://drive.google.com/drive/folders/1ePzTjdMPy7Snm47I1AQrpq3GrwhSkV7I?usp=sharing
- CLAMP_hand/data/hands/data : https://drive.google.com/drive/folders/1CZC0uRSmgHpNXFwXPBRJGcnu7NxWmO6A
- CLAMP_hand/work_dirs/CLAMP_ViTB_ap10k_256x256_onehand : https://drive.google.com/drive/folders/1MrIAbRX7tWOdeQNdS5jE2Q11cNr6wMjY?usp=sharing
- CLAMP_hand/work_dirs/CLAMP_ViTB_ap10k_256x256 : https://drive.google.com/drive/folders/1C2HPKCMDFfFyP9gCTL4Be3v096RW8Y0W?usp=sharing
- CLAMP_hand/work_dirs/CLAMP_ViTB_ap10k_256x256_twohands : https://drive.google.com/drive/folders/1-2BbjSIbW70QIowjCjUInLCif8b9YCQl?usp=sharing
- pretrained : https://huggingface.co/sentence-transformers/clip-ViT-B-16

## Usage

### Training

**Training CLAMP on AP-10K**

```
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 4 "0,1,2,3"
```

### Evaluation

You can get the pretrained model (the link is in "Main Results" session), then run following command to evaluate it on the validation set:

```
bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py work_dirs/CLAMP_ViTB_ap10k_256x256/epoch_210.pth 4 "0,1,2,3"
```

## Acknowledgement

This project is based on [mmpose](https://github.com/open-mmlab/mmpose), [AP-10K](https://github.com/AlexTheBad/AP-10K), [CLIP](https://github.com/openai/CLIP), and [DenseCLIP](https://github.com/raoyongming/DenseCLIP). Thanks for their wonderful works. See [LICENSE](https://github.com/Hand-CLAMP/hand-clamp/blob/main/LICENSE) for more details.

## Citing CLAMP

If you find CLAMP useful in your research, please consider citing:

```bash
@inproceedings{zhang2023clamp,
  title={CLAMP: Prompt-Based Contrastive Learning for Connecting Language and Animal Pose},
  author={Zhang, Xu and Wang, Wen and Chen, Zhe and Xu, Yufei and Zhang, Jing and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23272--23281},
  year={2023}
}
```
