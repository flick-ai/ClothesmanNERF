# ClothesmanNERF

这是我们CV作业的github储存仓库，用于实现从两个视频中提取人物进行三维建模并进行姿势与服装的迁移。

## Prerequisite

### `Configure environment`

请确认您已经配置好我们所要求的环境，环境需求之后会补充到下文。
如要成功运行本代码，需准备好四个conda环境：vibe, detectron, humannerf, must。 前两个环境将用来数据集生成， 后两个环境用来运行我们的代码。

### `Dataset Generation`

#### `data preparation`
请准备好两段视频：Source Video和Target Video我们的工作将实现将Target Video中的人物姿态和服装迁移到Source Video中的人物上。

#### `run pipline`

## Run on ZJU-Mocap Dataset

Below we take the subject 387 as a running example.

### `Prepare a dataset`

First, download ZJU-Mocap dataset from [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset).

Second, modify the yaml file of subject 387 at `tools/prepare_zju_mocap/387.yaml`. In particular,  `zju_mocap_path` should be the directory path of the ZJU-Mocap dataset.

```
dataset:
    zju_mocap_path: /path/to/zju_mocap
    subject: '387'
    sex: 'neutral'

...
```

Finally, run the data preprocessing script.

    cd tools/prepare_zju_mocap
    python prepare_dataset.py --cfg 387.yaml
    cd ../../

### `Train/Download models`

Now you can either download a pre-trained model by running the script.

    ./scripts/download_model.sh 387

or train a model by yourself. We used 4 GPUs (NVIDIA RTX 2080 Ti) to train a model.

    python train.py --cfg configs/human_nerf/zju_mocap/387/adventure.yaml

For sanity check, we provide a configuration that supports training on a single GPU (NVIDIA RTX 2080 Ti). Notice the performance is not guranteed for this configuration.

    python train.py --cfg configs/human_nerf/zju_mocap/387/single_gpu.yaml

### `Render output`

Render the frame input (i.e., observed motion sequence).

    python run.py \
        --type movement \
        --cfg configs/human_nerf/zju_mocap/387/adventure.yaml

Run free-viewpoint rendering on a particular frame (e.g., frame 128).

    python run.py \
        --type freeview \
        --cfg configs/human_nerf/zju_mocap/387/adventure.yaml \
        freeview.frame_idx 128


Render the learned canonical appearance (T-pose).

    python run.py \
        --type tpose \
        --cfg configs/human_nerf/zju_mocap/387/adventure.yaml

In addition, you can find the rendering scripts in `scripts/zju_mocap`.


## Run on a Custom Monocular Video

To get the best result, we recommend a video clip that meets these requirements:

- The clip has less than 600 frames (~20 seconds).
- The human subject shows most of body regions (e.g., front and back view of the body) in the clip.

### `Prepare a dataset`

To train on a monocular video, prepare your video data in `dataset/wild/monocular` with the following structure:

    monocular
        ├── images
        │   └── ${item_id}.png
        ├── masks
        │   └── ${item_id}.png
        └── metadata.json

We use `item_id` to match a video frame with its subject mask and metadata. An `item_id` is typically some alphanumeric string such as `000128`.

#### **images**

A collection of video frames, stored as PNG files.

#### **masks**

A collection of subject segmentation masks, stored as PNG files.

#### **metadata.json**

This json file contains metadata for video frames, including:

- human body pose (SMPL poses and betas coefficients)
- camera pose (camera intrinsic and extrinsic matrices). We follow [OpenCV](https://learnopencv.com/geometry-of-image-formation/) camera coordinate system and use [pinhole camera model](https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/CV/PinholeCamera/PinholeCamera.html).

You can run SMPL-based human pose detectors (e.g., [SPIN](https://github.com/nkolot/SPIN), [VIBE](https://github.com/mkocabas/VIBE), or [ROMP](https://github.com/Arthur151/ROMP)) on a monocular video to get body poses as well as camera poses.


```javascript
{
  // Replace the string item_id with your file name of video frame.
  "item_id": {
        // A (72,) array: SMPL coefficients controlling body pose.
        "poses": [
            -3.1341, ..., 1.2532
        ],
        // A (10,) array: SMPL coefficients controlling body shape.
        "betas": [
            0.33019, ..., 1.0386
        ],
        // A 3x3 camera intrinsic matrix.
        "cam_intrinsics": [
            [23043.9, 0.0,940.19],
            [0.0, 23043.9, 539.23],
            [0.0, 0.0, 1.0]
        ],
        // A 4x4 camera extrinsic matrix.
        "cam_extrinsics": [
            [1.0, 0.0, 0.0, -0.005],
            [0.0, 1.0, 0.0, 0.2218],
            [0.0, 0.0, 1.0, 47.504],
            [0.0, 0.0, 0.0, 1.0],
        ],
  }

  ...

  // Iterate every video frame.
  "item_id": {
      ...
  }
}
```

Once the dataset is properly created, run the script to complete dataset preparation.

    cd tools/prepare_wild
    python prepare_dataset.py --cfg wild.yaml
    cd ../../

### `Train a model`

Now we are ready to lanuch a training. By default, we used 4 GPUs (NVIDIA RTX 2080 Ti) to train a model.

    python train.py --cfg configs/human_nerf/wild/monocular/adventure.yaml

For sanity check, we provide a single-GPU (NVIDIA RTX 2080 Ti) training config. Note the performance is not guaranteed for this configuration.

    python train.py --cfg configs/human_nerf/wild/monocular/single_gpu.yaml

### `Render output`

Render the frame input (i.e., observed motion sequence).

    python run.py \
        --type movement \
        --cfg configs/human_nerf/wild/monocular/adventure.yaml

Run free-viewpoint rendering on a particular frame (e.g., frame 128).

    python run.py \
        --type freeview \
        --cfg configs/human_nerf/wild/monocular/adventure.yaml \
        freeview.frame_idx 128


Render the learned canonical appearance (T-pose).

    python run.py \
        --type tpose \
        --cfg configs/human_nerf/wild/monocular/adventure.yaml

In addition, you can find the rendering scripts in `scripts/wild`.

## Acknowledgement

The implementation took reference from [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch), [Neural Body](https://github.com/zju3dv/neuralbody), [Neural Volume](https://github.com/facebookresearch/neuralvolumes), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), and [YACS](https://github.com/rbgirshick/yacs). We thank the authors for their generosity to release code.

## Citation

If you find our work useful, please consider citing:

```BibTeX
@InProceedings{weng_humannerf_2022_cvpr,
    title     = {Human{N}e{RF}: Free-Viewpoint Rendering of Moving People From Monocular Video},
    author    = {Weng, Chung-Yi and
                 Curless, Brian and
                 Srinivasan, Pratul P. and
                 Barron, Jonathan T. and
                 Kemelmacher-Shlizerman, Ira},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {16210-16220}
}
```