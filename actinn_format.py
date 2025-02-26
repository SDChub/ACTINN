import numpy as np
import pandas as pd
import scipy.io
import os
import argparse

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to the input file or the 10X directory.")
    parser.add_argument("-o", "--output", type=str, help="Prefix of the output file.")
    parser.add_argument("-f", "--format", type=str, help="Format of the input file (10X_V2, 10X_V3, txt, csv).")
    return parser

if __name__ == '__main__':
    # 获取命令行参数解析器
    parser = get_parser()
    # 解析命令行参数
    args = parser.parse_args()
    if args.format == "10X_V2":
        # 获取输入路径
        path = args.input
        # 确保路径以"/"结尾
        if path[-1] != "/":
            path += "/"
        # 读取10X格式的稀疏矩阵文件
        new = scipy.io.mmread(os.path.join(path, "matrix.mtx"))
        # 读取基因信息文件，获取基因名列表
        genes = list(pd.read_csv(path+"genes.tsv", header=None, sep='\t')[1])
        # 读取细胞条形码文件，获取细胞标识符列表
        barcodes = list(pd.read_csv(path+"barcodes.tsv", header=None)[0])
        # 将稀疏矩阵转换为密集矩阵，并创建DataFrame，行索引为基因名，列索引为细胞条形码
        new = pd.DataFrame(np.array(new.todense()), index=genes, columns=barcodes)
        # 将NaN值填充为0
        new.fillna(0, inplace=True)
        # 获取唯一基因名的索引位置，去除重复的基因名
        uniq_index = np.unique(new.index, return_index=True)[1]
        # 使用唯一索引筛选数据，去除重复基因
        new = new.iloc[uniq_index,]
        # 移除所有表达量为零的基因（行和为0的行）
        new = new.loc[new.sum(axis=1)>0, :]
        # 打印处理后矩阵的维度信息
        print("Dimension of the matrix after removing all-zero rows:", new.shape)
        # 将处理后的数据保存为HDF5格式，使用"dge"作为键名，压缩级别为3
        new.to_hdf(args.output+".h5", key="dge", mode="w", complevel=3)

    if args.format == "10X_V3":
        # 获取输入路径
        path = args.input
        # 确保路径以"/"结尾
        if path[-1] != "/":
            path += "/"
        # 读取10X V3格式的压缩稀疏矩阵文件
        new = scipy.io.mmread(os.path.join(path, "matrix.mtx.gz"))
        # 读取特征信息文件，获取基因名列表（第2列）
        genes = list(pd.read_csv(path+"features.tsv.gz", header=None, sep='\t')[1])
        # 读取细胞条形码文件，获取细胞标识符列表（第1列）
        barcodes = list(pd.read_csv(path+"barcodes.tsv.gz", header=None)[0])
        # 将稀疏矩阵转换为密集矩阵，并创建DataFrame，行索引为基因名，列索引为细胞条形码
        new = pd.DataFrame(np.array(new.todense()), index=genes, columns=barcodes)
        # 将NaN值填充为0
        new.fillna(0, inplace=True)
        # 获取唯一基因名的索引位置，去除重复的基因名
        uniq_index = np.unique(new.index, return_index=True)[1]
        # 使用唯一索引筛选数据，去除重复基因
        new = new.iloc[uniq_index,]
        # 移除所有表达量为零的基因（行和为0的行）
        new = new.loc[new.sum(axis=1)>0, :]
        # 打印处理后矩阵的维度信息
        print("Dimension of the matrix after removing all-zero rows:", new.shape)
        # 将处理后的数据保存为HDF5格式，使用"dge"作为键名，压缩级别为3
        new.to_hdf(args.output+".h5", key="dge", mode="w", complevel=3)

    if args.format == "csv":
        new = pd.read_csv(args.input, index_col=0)
        uniq_index = np.unique(new.index, return_index=True)[1]
        new = new.iloc[uniq_index,]
        new = new.loc[new.sum(axis=1)>0, :]
        print("Dimension of the matrix after removing all-zero rows:", new.shape)
        new.to_hdf(args.output+".h5", key="dge", mode="w", complevel=3)

    if args.format == "txt":
        new = pd.read_csv(args.input, index_col=0, sep="\t")
        uniq_index = np.unique(new.index, return_index=True)[1]
        new = new.iloc[uniq_index,]
        new = new.loc[new.sum(axis=1)>0, :]
        print("Dimension of the matrix after removing all-zero rows:", new.shape)
        new.to_hdf(args.output+".h5", key="dge", mode="w", complevel=3)
