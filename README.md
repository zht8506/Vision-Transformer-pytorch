### 关于项目

本项目代码主要来源于

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer

在此基础上进行了书写了一些注释。

```
  ├── vit_model.py: ViT模型搭建
  ├── weights: 权重文件保存的文件夹
  ├── train.py: 训练脚本
  ├── predict.py: 单张图像预测脚本
  ├── my_dataset.py: 重写dataset类，用于读取数据集
  ├── flops.py: 计算浮点量的代码
  └── utils.py：本项目涉及的常用操作的代码
```

### 训练

在train.py脚本下，opt选择设置数据集和权重路径，设置batch_size等参数。

### 相关下载

本项目使用的是花分类数据集，首先需要下载花分类数据集，链接为

http://download.tensorflow.org/example_images/flower_photos.tgz

预训练权重下载

https://github.com/google-research/vision_transformer

本项目使用的pytorch版本的权重下载，其他的权重在vit_model.py上有下载链接

https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth

个人关于ViT解读的文章

https://zhuanlan.zhihu.com/p/461077472
