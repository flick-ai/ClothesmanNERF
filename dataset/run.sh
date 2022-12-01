#!/bin/bash

# 自定义视频id
data_id=nilu

# 给定数据集参数
dataset_path=./dataset/video/
data_name=$data_id.mp4
target_folder=./dataset/origin/$data_id
mkdir -p $target_folder

# SMPL参数估计
conda activate vibe # vibe 所需conda环境
cd VIBE # vibe 路径
python demo.py --vid_file $dataset_path$data_name --output_folder output/ --image_folder $target_folder/images --sideview
python process.py --data_id $data_id --output_path $target_folder/
conda deactivate

# Mask输出
conda activate vibe-env # detectron2所需conda环境
cd /home/wenhao/CV/detectron2/demo # detectron2 路径
mkdir -p $target_folder/masks
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input $target_folder/images/*.png --output $target_folder/masks  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl
conda deactivate

# 格式转换
cd /Cell/ClothesmanNERF/ # 项目目录 路径
python prepare_dataset.py --subject $data_id --path ./dataset/origin
cp -r $target_folder ./dataset/wild/$data_id