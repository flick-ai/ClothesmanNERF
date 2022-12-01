# ClothesmanNERF

这是我们CV作业的github储存仓库，用于实现从两个视频中提取人物进行三维建模并进行姿势与服装的迁移。

## Prerequisite

### `Configure environment`

请确认您已经配置好我们所要求的环境，环境需求之后会补充到下文。
如要成功运行本代码，需准备好四个conda环境：vibe, detectron, humannerf, must。 前两个环境将用来数据集生成， 后两个环境用来运行我们的代码。关于前两个环境的搭建，我们强烈建议您git clone前两个开源仓库的代码并安装所需环境。

### `Dataset Generation`

#### `data preparation`
请准备好两段视频：Source Video和Target Video我们的工作将实现将Target Video中的人物姿态和服装迁移到Source Video中的人物上。
首先将两段视频放置到`./dataset/video`目录下。然后编辑`./dataset/run.sh`填写您的环境信息并运行他，即可得到我们所要求格式的数据集。
这里我们建议的视频长度尽量控制在30s左右。

## Train Model

## Acknowledgement

我们的实现参考了 [HumanNERF](https://github.com/chungyiweng/humannerf), [Neuman](https://github.com/apple/ml-neuman),和[MUST-GAN](https://github.com/TianxiangMa/MUST-GAN). 我们十分感谢作者开源了他们的项目代码。