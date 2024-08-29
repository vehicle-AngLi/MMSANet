Hello, this is a pre-create package for my paper, I will upload my codes and pre-trained models after acceptance, See you ~

2024.05.31: Now some examples have been uploaded, welcome to check!

2024.08.27: HI! I have to appreciate the high efficiency of IEEE T-IV, our EA version of paper now has been published. However, we are arrange and develop our codes, which will be avaliable before 2024.09.01!

2024.08.29：Now, our codes and pre-trained modle have been available! (I wanna pull them tomorrow, because I am going to have KFC......)

![back](https://github.com/user-attachments/assets/9a271061-9600-4e84-b514-87a3c0465941)

# Why Choose Our Method?

![two_as](https://github.com/user-attachments/assets/9fba7773-86ba-4eb6-ae0a-b226566b6548)

A little heavier, Much more Accurate, Expecially for Pedestrians and Partially-Features-Missed Targets.

![results2](https://github.com/user-attachments/assets/a8941e48-449a-4e12-8a26-289071e1bcc2)

# How to Use?

First you can use the **pip install -r requirements.txt** to build the Environment.

## Use our Pre-trained module?

First you need to prepare your test datasets. Place your visible images in **/test_imgs/vi/**, and infrared images (if you have) in **/test_imgs/ir/**.

If your datasets are single source, run:
**python detect.py --weights pre_trained/visible/vip_lit.pt --fuse 0**

Here we prepared different pre-trained modules in **pre_trained/visible/**, which are trained in KITTI datasets and LLVIP datasets. Please choose them properly. If your datasets are self-collected, we recommend you to train a new module.

If your datasets have infrared source, run:
**python detect.py --weights pre_trained/bimodal_vip/lit.pt --fuse 1**

Here we also prepared different pre-trained modules in **pre_trained/bimodal_vip/**.

## Train Your MMSA detectors?

First of all, you need to prepare one document of YAML, to state your datasets. Here we named it as data.yaml

Thewn RUN:

**python train.py --weights yolov5s.pt --data data/data.yaml --cfg models/MMSANet.yaml --batch-size ... --epochs ...**

We prepared light version in /models, and the epochs or batch size are depend on your own need!

# Acknowledgements

Yolov5: https://github.com/ultralytics/yolov5

RCAFusion: https://github.com/vehicle-AngLi/RCAFusion

Datasets - LLVIP: https://github.com/bupt-ai-cz/LLVIP

Datasets - KITTI: https://www.cvlibs.net/datasets/kitti

# Cite Us

@ARTICLE{10648743,

  author={Li, Ang and Wang, Ziwei and Wang, Fanxun and Liu, Zhichao and Yin, Guodong and Fang, Ruiqi and Geng, Keke},
  
  journal={IEEE Transactions on Intelligent Vehicles}, 
  
  title={A Novel Semantic Information Perception Architecture for Extreme Targets Detection in Complex Traffic Scenarios}, 
  
  year={2024},
  
  volume={},
  
  number={},
  
  pages={1-13},
  
  keywords={Feature extraction;Object detection;Autonomous vehicles;Lighting;Computer architecture;Task analysis;Semantics;Bimodal perception;Complex traffic scenarios;Mixed multi-scales network;Vision-based object detection},
  
  doi={10.1109/TIV.2024.3450201}}
