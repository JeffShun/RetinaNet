## 一、模型介绍
基于RetinaNet的目标检测框架，backbone和损失函数等自定义

## 二、文件结构说明

### 训练文件目录

- train/train.py: 单卡训练代码入口
- train/train_multi_gpu.py: 分布式训练代码入口
- train/custom/dataset/dataset.py: dataset类
- train/custom/model/decoder.py: 预测结果解码器
- train/custom/model/anchor.py: anchor生成
- train/custom/model/neck.py: 模型neck
- train/custom/model/head.py: 模型head
- train/custom/model/loss.py 模型loss
- train/custom/model/backbones/*.py 生成网络backbone
- train/custom/model/network.py: 网络整体框架
- train/custom/utils/*.py 训练相关工具函数
- train/config/model_config.py: 训练的配置文件

### 预测文件目录

* test/test_config.yaml: 预测配置文件
* test/main.py: 预测入口文件
* test/predictor.py: 模型预测具体实现，包括加载模型和后处理
* test/analysis_tools/*.py 结果分析工具函数，如计算评估指标

## 三、demo调用方法

1. 准备训练原始数据
   * 在train文件夹下新建train_data/origin_data文件夹，放入训练的原始训练数据

2. 生成处理后的训练数据，在train_data/processed_data文件夹下
   * cd train
   * python custom/utils/generate_dataset.py

3. 开始训练
   * 分布式训练：sh ./train_dist.sh
   * 单卡训练：python train.py
   
4. 准备测试数据
   * 将预测数据放入test/data/input目录

5. 开始预测
   * cd test
   * python main.py

6. 结果评估
   * python test/analysis_tools/cal_matrics.py
