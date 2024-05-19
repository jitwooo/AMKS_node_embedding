# 基于平均混合核特征的图节点结构嵌入算法研究

## 1 数据集

数据集下载链接：https://pan.baidu.com/s/1sbURWpThNiQ7oT6tKLvz5w?pwd=gkjg 

下载完成后，解压保存在根目录下。

## 2 环境

- 实验运行操作系统为 `Ubuntu20.04`，`CUDA` 版本为12.3。

- `conda` 环境保存在文件 `environment.yml`中，运行命令 `conda env create -f environment.yml` 进行安装，环境名为 `AMKS`。
- 修改代码运行根目录：进入文件 `src/running/rw_directory` 中，修改 `root_path` 为本地运行根目录，`data_path` 为数据根目录。

## 3 代码说明

### src

`src` 中存放模型源代码和实验工具代码。

- `experiment_utils` 和 `running` ：存放运行实验所需工具代码。
- `model`：存放实验模型源代码，包含`AMKS`、`GraphWave` 和 `node2vec` 三种模型。

### experiment

`experiment` 中存放各实验运行代码。

- `ex1_eigenvectors.ipynb`

  论文4.2节：图4-6拉普拉斯矩阵与频谱信息的对应关系

- `ex2_time_chosing.ipynb`

  论文4.1.1节：时间参数的选择

- `ex3_energy_level_chosing.ipynb`

  论文4.1.2节：能级参数的选择

- `ex4_graph_disturbance.ipynb`

  论文4.3.1节：图形扰动实验

- `ex5_eigenvectors_mixing.ipynb`

  论文4.2节：拉普拉斯特征空间混合实验

- `ex8`

  论文4.3.2节：单图相似结构节点分类实验

- `ex9`

  论文4.3.3节：跨图相似结构节点匹配实验

- `*.sh`

  运行部分实验命令示例
