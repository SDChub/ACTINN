# Automated Cell Type Identification using Neural Networks
# 使用神经网络的自动化细胞类型识别

## Overview
ACTINN (Automated Cell Type Identification using Neural Networks) is a bioinformatic tool to quickly and accurately identify cell types in scRNA-Seq. For details, please read the paper:
## 概述
ACTINN (使用神经网络的自动化细胞类型识别)是一个生物信息学工具，用于在单细胞RNA测序(scRNA-Seq)中快速准确地识别细胞类型。详细信息请参阅论文：
https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btz592/5540320

All datasets used in the paper are available here:
论文中使用的所有数据集可在此处获取：
https://figshare.com/articles/ACTINN/8967116

## Prerequisite
## 前提条件
python 3.6

python packages:
python 包要求：
tensorflow 1.10+, numpy 1.14+, pandas 0.23+, argparse 1.1+, scipy 1.1+

## Convert format
We use HDF5 format for the scRNA-Seq expressiong matrix, which stores the compressed matrix and is fast to load. To convert the format, we first read the expression matrix as a pandas dataframe, then we use the to_hdf function to save the file as HDF5 format. For the to_hdf function, we use "dge", which stands for digital gene expression, for the key parameter.
## 格式转换
我们使用HDF5格式存储scRNA-Seq表达矩阵，该格式可以存储压缩矩阵并且加载速度快。要转换格式，我们首先将表达矩阵读取为pandas数据框，然后使用to_hdf函数将文件保存为HDF5格式。对于to_hdf函数，我们使用"dge"(数字基因表达的缩写)作为key参数。

### Usage
### 使用方法
```
python actinn_format.py -i input_file -o output_prefix -f format
python actinn_format.py -i 输入文件 -o 输出前缀 -f 格式
```

### Parameters
* -i	Path to the input file or the 10X directory
* -o	Prefix of the output file
* -f	Format of the input file (10X_V2, 10X_V3, txt, csv)
### 参数
* -i    输入文件或10X目录的路径
* -o    输出文件的前缀
* -f    输入文件格式(10X_V2, 10X_V3, txt, csv)

### Output
The output will be an HDF5 formated file named after the output prefix with ".h5" extension
### 输出
输出将是一个HDF5格式的文件，文件名为输出前缀加上".h5"扩展名

### Examples
### 示例

#### Convert 10X_V2 format
#### 转换10X_V2格式
```
python actinn_format.py -i ./test_data/train_set_10x -o train_set -f 10X_V2
```

#### Convert 10X_V3 format
#### 转换10X_V3格式
```
python actinn_format.py -i ./test_data/train_set_10x -o train_set -f 10X_V3
```

#### Convert txt format
#### 转换txt格式
```
python actinn_format.py -i ./test_data/train_set.txt.gz -o train_set -f txt
```

#### Convert csv format
#### 转换csv格式
```
python actinn_format.py -i ./test_data/train_set.csv.gz -o train_set -f csv
```

## Predict cell types
We train a 4 layer (3 hidden layers) neural network on scRNA-Seq datasets with predifined cell types, then we use the trained parameters to predict cell types for other datasets.
## 预测细胞类型
我们在具有预定义细胞类型的scRNA-Seq数据集上训练一个4层(3个隐藏层)神经网络，然后使用训练好的参数来预测其他数据集的细胞类型。

### Usage
### 使用方法
```
python actinn_predict.py -trs training_set -trl training_label -ts test_set -lr learning_rat -ne num_epoch -ms minibatch_size -pc print_cost -op output_probability
python actinn_predict.py -trs 训练集 -trl 训练标签 -ts 测试集 -lr 学习率 -ne 训练轮数 -ms 小批量大小 -pc 打印代价 -op 输出概率
```

### Parameters
* -trs	Path to the training set, must be HDF5 format with key "dge".
* -trl	Path to the training label (the cell types for the training set), must be tab separated text file with no column and row names.
* -ts	Path to test sets, must be HDF5 format with key "dge".
* -lr	Learning rate (default: 0.0001). We can increase the learning rate if the cost drops too slow, or decrease the learning rate if the cost drops super fast in the beginning and starts to fluctuate in later epochs.
* -ne	Number of epochs (default: 50). The number of epochs can be determined by looking at the cost after each epoch. If the cost starts to decrease very slowly after ceartain epoch, then the "ne" parameter should be set to that epoch number. 
* -ms	Minibatch size (default: 128). This parameter can be set larger when training a large dataset.
* -pc	Print cost (default: True). Whether to print cost after each 5 epochs.
* -op Output probabilities for each cell being the cell types in the training data (default: False).
### 参数
* -trs  训练集路径，必须是带有"dge"键的HDF5格式
* -trl  训练标签路径(训练集的细胞类型)，必须是无列名和行名的制表符分隔文本文件
* -ts   测试集路径，必须是带有"dge"键的HDF5格式
* -lr   学习率(默认：0.0001)。如果代价下降太慢可以增加学习率，如果代价在开始时下降太快而后期开始波动则可以降低学习率
* -ne   训练轮数(默认：50)。训练轮数可以通过观察每轮后的代价来确定。如果代价在某轮后开始非常缓慢地下降，那么"ne"参数应该设置为该轮数
* -ms   小批量大小(默认：128)。在训练大型数据集时可以将此参数设置得更大
* -pc   打印代价(默认：True)。是否在每5轮后打印代价
* -op   输出训练数据中每个细胞作为各种细胞类型的概率(默认：False)

### Output
The output will be a file named "predicted_label.txt". In the file, the first column will be the cell name, the second column will be the predicted cell type. 
If the "op" parameter is set to True, there will be another output file named "predicted_probablities.txt", where columns are cells and rows are cell types. The number in row i and column j will be the probablity that cell j being cell type i.
### 输出
输出将是一个名为"predicted_label.txt"的文件。在该文件中，第一列是细胞名称，第二列是预测的细胞类型。
如果"op"参数设置为True，将会有另一个名为"predicted_probablities.txt"的输出文件，其中列是细胞，行是细胞类型。第i行第j列的数字表示第j个细胞作为第i种细胞类型的概率。

### Example
### 示例
```
python actinn_predict.py -trs ./test_data/train_set.h5 -trl ./test_data/train_label.txt.gz -ts ./test_data/test_set.h5 -lr 0.0001 -ne 50 -ms 128 -pc True -op False
```

## Plots
We show an example on how to create a tSNE plot with the predicted cell types. The R command can be found in the "tSNE_Example" folder.
## 图表
我们展示了如何使用预测的细胞类型创建tSNE图的示例。R命令可以在"tSNE_Example"文件夹中找到。
![tSNE Plot](https://github.com/mafeiyang/ACTINN/blob/master/tSNE_Example/tSNE_Plot.png)
![tSNE图](https://github.com/mafeiyang/ACTINN/blob/master/tSNE_Example/tSNE_Plot.png)
