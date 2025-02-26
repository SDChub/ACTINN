# 导入所需的Python包
import numpy as np
import pandas as pd
import sys
import math
import collections
import tensorflow as tf
import argparse
import timeit
run_time = timeit.default_timer()
from tensorflow.python.framework import ops

# 定义命令行参数解析器函数
def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument("-trs", "--train_set", type=str, help="Training set file path.")
    parser.add_argument("-trl", "--train_label", type=str, help="Training label file path.")
    parser.add_argument("-ts", "--test_set", type=str, help="Training set file path.")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate (default: 0.0001)", default=0.0001)
    parser.add_argument("-ne", "--num_epochs", type=int, help="Number of epochs (default: 50)", default=50)
    parser.add_argument("-ms", "--minibatch_size", type=int, help="Minibatch size (default: 128)", default=128)
    parser.add_argument("-pc", "--print_cost", type=bool, help="Print cost when training (default: True)", default=True)
    parser.add_argument("-op", "--output_probability", type=bool, help="Output the probabilities for each cell being the cell types in the training data (default: False)", default=False)
    return parser

# 数据预处理函数：获取共同基因，标准化和缩放数据集
def scale_sets(sets):
    # 输入 -- 一个包含所有需要缩放的数据集的列表
    # 输出 -- 经过缩放处理的数据集
    
    # 第一步：找出所有数据集中共同存在的基因
    # 首先获取第一个数据集的基因列表
    common_genes = set(sets[0].index)
    # 通过集合的交集操作，找出所有数据集中都存在的基因
    for i in range(1, len(sets)):
        common_genes = set.intersection(set(sets[i].index), common_genes)
    # 将共同基因集合转换为排序列表，便于后续处理
    common_genes = sorted(list(common_genes))
    
    # 记录每个数据集的分割点，用于后续从合并数据集中分离出各个数据集
    sep_point = [0]
    
    # 第二步：对每个数据集进行处理，只保留共同基因
    for i in range(len(sets)):
        # 使用loc索引器选择共同基因的行
        sets[i] = sets[i].loc[common_genes,]
        # 记录每个数据集的列数（细胞数量），用于后续分割
        sep_point.append(sets[i].shape[1])
    
    # 第三步：数据标准化处理
    # 将所有数据集按列合并成一个大矩阵，并转换为float32类型
    total_set = np.array(pd.concat(sets, axis=1, sort=False), dtype=np.float32)
    # 对每个细胞（列）进行归一化处理，使每列总和为10000（CPM标准化）
    total_set = np.divide(total_set, np.sum(total_set, axis=0, keepdims=True)) * 10000
    # 对数据进行log2(x+1)转换，减小数据范围差异
    total_set = np.log2(total_set+1)
    
    # 第四步：过滤表达量异常的基因
    # 计算每个基因的总表达量
    expr = np.sum(total_set, axis=1)
    # 保留表达量在1%到99%百分位数之间的基因，过滤掉表达量过高或过低的基因
    total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
    # 计算每个基因的变异系数（CV = 标准差/均值）
    cv = np.std(total_set, axis=1) / np.mean(total_set, axis=1)
    # 保留变异系数在1%到99%百分位数之间的基因，过滤掉变异过大或过小的基因
    total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
    
    # 第五步：将处理后的大矩阵分割回各个数据集
    for i in range(len(sets)):
        # 使用分割点信息，从大矩阵中提取对应的列，重新组成各个数据集
        sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]
    
    # 返回处理后的数据集列表
    return sets

# 将标签转换为独热编码矩阵
def one_hot_matrix(labels, C):
    # 输入参数:
    # - labels: 数据集的真实标签列表
    # - C: 类别总数
    # 输出:
    # - 形状为(类别数, 样本数)的独热编码矩阵
    
    # 将C转换为TensorFlow常量，并命名为"C"
    C = tf.constant(C, name = "C")
    
    # 使用tf.one_hot函数将标签转换为独热编码
    # axis=0表示独热向量沿着第0轴展开(即每列是一个独热向量)
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    
    # 创建TensorFlow会话
    sess = tf.Session()
    
    # 在会话中运行one_hot_matrix操作，获取实际的独热编码矩阵
    one_hot = sess.run(one_hot_matrix)
    
    # 关闭会话，释放资源
    sess.close()
    
    # 返回生成的独热编码矩阵
    return one_hot

# 创建细胞类型到标签的映射字典
def type_to_label_dict(types):
    # 输入参数: types - 包含所有细胞类型的列表
    # 输出: 将细胞类型映射到数值标签的字典
    
    # 初始化一个空字典，用于存储细胞类型到标签的映射
    type_to_label_dict = {}
    
    # 使用set()函数去除重复的细胞类型，然后转换回列表
    # 这样可以获得所有唯一的细胞类型
    all_type = list(set(types))
    
    # 遍历所有唯一的细胞类型
    for i in range(len(all_type)):
        # 为每种细胞类型分配一个唯一的数字标签(从0开始)
        # 例如: {'CD4 T cell': 0, 'CD8 T cell': 1, 'B cell': 2, ...}
        type_to_label_dict[all_type[i]] = i
    
    # 返回构建好的映射字典
    return type_to_label_dict

# 将细胞类型转换为数值标签
def convert_type_to_label(types, type_to_label_dict):
    # 输入参数:
    # - types: 细胞类型列表
    # - type_to_label_dict: 细胞类型到数值标签的映射字典
    # 输出:
    # - 转换后的数值标签列表
    
    # 确保types是列表类型
    types = list(types)
    
    # 初始化一个空列表，用于存储转换后的数值标签
    labels = list()
    
    # 遍历每个细胞类型
    for type in types:
        # 使用映射字典将细胞类型转换为对应的数值标签，并添加到labels列表中
        labels.append(type_to_label_dict[type])
    
    # 返回转换后的数值标签列表
    return labels

# 创建神经网络的占位符
def create_placeholders(n_x, n_y):
    """
    创建神经网络的输入占位符
    
    这段代码的作用是为神经网络创建输入和输出的占位符。在TensorFlow中，占位符是用来接收外部数据的特殊变量，
    它们在神经网络的训练和预测过程中起着至关重要的作用。
    
    在训练中，这些占位符允许我们将基因表达数据(X)和细胞类型标签(Y)输入到计算图中，而不需要在图构建时就提供实际数据。
    这使得我们可以使用相同的计算图处理不同批次的数据，实现小批量梯度下降等训练技术。
    
    参数:
    n_x -- 输入特征的数量（基因数量）
    n_y -- 输出类别的数量（细胞类型数量）
    
    返回:
    X -- 输入数据的占位符，形状为(n_x, None)，其中None表示批次大小可变
    Y -- 标签数据的占位符，形状为(n_y, None)，用于存储独热编码的细胞类型标签
    """
    # 创建输入数据X的占位符，数据类型为float32，形状为(特征数, 样本数)
    X = tf.placeholder(tf.float32, shape = (n_x, None))
    # 创建标签数据Y的占位符，数据类型为float32，形状为(类别数, 样本数)
    Y = tf.placeholder(tf.float32, shape = (n_y, None))
    # 返回创建的两个占位符
    return X, Y

# 初始化神经网络参数
def initialize_parameters(nf, ln1, ln2, ln3, nt):
    """
    初始化神经网络的参数（权重和偏置）
    
    参数:
    nf -- 输入特征的数量（基因数量）
    ln1 -- 第一个隐藏层的节点数
    ln2 -- 第二个隐藏层的节点数
    ln3 -- 第三个隐藏层的节点数
    nt -- 输出类别的数量（细胞类型数量）
    
    返回:
    parameters -- 包含所有权重和偏置的字典
    """
    # 设置随机种子为3，确保每次运行得到相同的初始化参数，使结果可重复
    tf.set_random_seed(3)
    
    # 使用Xavier初始化方法初始化第一层的权重矩阵，形状为[ln1, nf]
    # Xavier初始化有助于防止深层网络中的梯度消失或爆炸问题
    W1 = tf.get_variable("W1", [ln1, nf], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b1 = tf.get_variable("b1", [ln1, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [ln2, ln1], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b2 = tf.get_variable("b2", [ln2, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [ln3, ln2], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b3 = tf.get_variable("b3", [ln3, 1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [nt, ln3], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b4 = tf.get_variable("b4", [nt, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}
    return parameters

# 前向传播函数
def forward_propagation(X, parameters):
    """
    实现神经网络的前向传播过程
    
    模型结构: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    参数:
    X -- 输入数据集，形状为(特征数量, 样本数量)
    parameters -- 包含网络参数的字典: "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"
    
    返回:
    Z4 -- 最后一个线性单元的输出，用于后续的softmax计算
    """
    # 从参数字典中获取第一层的权重和偏置
    W1 = parameters['W1']
    b1 = parameters['b1']
    # 从参数字典中获取第二层的权重和偏置
    W2 = parameters['W2']
    b2 = parameters['b2']
    # 从参数字典中获取第三层的权重和偏置
    W3 = parameters['W3']
    b3 = parameters['b3']
    # 从参数字典中获取第四层的权重和偏置
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    # 前向计算第一层
    # 计算第一层的线性部分: Z1 = W1·X + b1
    Z1 = tf.add(tf.matmul(W1, X), b1)
    # 应用ReLU激活函数: A1 = ReLU(Z1)
    A1 = tf.nn.relu(Z1)
    
    # 前向计算第二层
    # 计算第二层的线性部分: Z2 = W2·A1 + b2
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    # 应用ReLU激活函数: A2 = ReLU(Z2)
    A2 = tf.nn.relu(Z2)
    
    # 前向计算第三层
    # 计算第三层的线性部分: Z3 = W3·A2 + b3
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    # 应用ReLU激活函数: A3 = ReLU(Z3)
    A3 = tf.nn.relu(Z3)
    
    # 前向计算第四层(输出层)
    # 计算第四层的线性部分: Z4 = W4·A3 + b4
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    
    # 返回最后一层的线性输出，后续会用于softmax计算
    return Z4

# 计算代价函数
def compute_cost(Z4, Y, parameters, lambd=0.01):
    """
    计算神经网络的代价函数
    
    参数:
    Z4 -- 前向传播的输出，形状为(类别数量, 样本数量)
    Y -- 真实标签，与Z4形状相同
    parameters -- 包含网络参数的字典: "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"
    lambd -- L2正则化参数，默认为0.01
    
    返回:
    cost -- 包含代价值的张量
    """
    # 转置Z4和Y，使其形状变为(样本数量, 类别数量)，符合softmax_cross_entropy_with_logits_v2的输入要求
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)
    
    # 计算交叉熵损失和L2正则化
    # 第一部分: 使用softmax交叉熵计算分类损失
    # 第二部分: 对所有权重矩阵应用L2正则化以防止过拟合
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels)) + \
    (tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"]) + tf.nn.l2_loss(parameters["W3"]) + tf.nn.l2_loss(parameters["W4"])) * lambd
    
    # 返回计算得到的总代价
    return cost

# 生成小批量数据
def random_mini_batches(X, Y, mini_batch_size=32, seed=1):
    """
    生成随机小批量数据用于训练
    
    参数:
    X -- 输入数据，形状为(特征数量, 样本数量)
    Y -- 标签数据，形状为(类别数量, 样本数量)
    mini_batch_size -- 每个小批量的样本数量，默认为32
    seed -- 随机数种子，用于复现结果，默认为1
    
    返回:
    mini_batches -- 包含(mini_batch_X, mini_batch_Y)元组的列表
    """
    # 获取样本数量
    ns = X.shape[1]
    # 初始化小批量列表
    mini_batches = []
    # 设置随机数种子以确保结果可复现
    np.random.seed(seed)
    
    # 生成随机排列，用于打乱数据
    permutation = list(np.random.permutation(ns))
    # 根据随机排列重新排序输入数据
    shuffled_X = X[:, permutation]
    # 同样方式打乱标签数据，保持与输入数据对应
    shuffled_Y = Y[:, permutation]
    
    # 计算可以完整分割的小批量数量
    num_complete_minibatches = int(math.floor(ns/mini_batch_size))
    
    # 循环创建完整的小批量
    for k in range(0, num_complete_minibatches):
        # 提取当前小批量的输入数据
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        # 提取当前小批量的标签数据
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        # 将输入和标签组合成一个小批量
        mini_batch = (mini_batch_X, mini_batch_Y)
        # 将小批量添加到列表中
        mini_batches.append(mini_batch)
    
    # 处理剩余的样本（如果样本数量不能被mini_batch_size整除）
    if ns % mini_batch_size != 0:
        # 提取剩余样本的输入数据
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : ns]
        # 提取剩余样本的标签数据
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : ns]
        # 将剩余样本组合成一个小批量
        mini_batch = (mini_batch_X, mini_batch_Y)
        # 将剩余样本的小批量添加到列表中
        mini_batches.append(mini_batch)
    
    # 返回所有小批量
    return mini_batches

# 用于预测的前向传播函数
def forward_propagation_for_predict(X, parameters):
    """
    用于预测的前向传播函数
    
    参数:
    X -- 用于预测的数据集
    parameters -- 训练后的模型参数
    
    返回:
    Z4 -- 最后一个线性单元的输出
    """
    # 从参数字典中提取第一层的权重和偏置
    W1 = parameters['W1']
    b1 = parameters['b1']
    # 从参数字典中提取第二层的权重和偏置
    W2 = parameters['W2']
    b2 = parameters['b2']
    # 从参数字典中提取第三层的权重和偏置
    W3 = parameters['W3']
    b3 = parameters['b3']
    # 从参数字典中提取第四层的权重和偏置
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    # 计算第一层的线性部分: Z1 = W1·X + b1
    Z1 = tf.add(tf.matmul(W1, X), b1)
    # 对Z1应用ReLU激活函数得到第一层的激活值A1
    A1 = tf.nn.relu(Z1)
    
    # 计算第二层的线性部分: Z2 = W2·A1 + b2
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    # 对Z2应用ReLU激活函数得到第二层的激活值A2
    A2 = tf.nn.relu(Z2)
    
    # 计算第三层的线性部分: Z3 = W3·A2 + b3
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    # 对Z3应用ReLU激活函数得到第三层的激活值A3
    A3 = tf.nn.relu(Z3)
    
    # 计算第四层(输出层)的线性部分: Z4 = W4·A3 + b4
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    
    # 返回输出层的线性输出(未经过softmax激活)
    return Z4

# 预测函数
def predict(X, parameters):
    """
    预测函数 - 使用训练好的模型参数对输入数据进行预测
    
    参数:
    X -- 用于预测的数据集
    parameters -- 训练后的模型参数
    
    返回:
    prediction -- 预测结果（类别索引）
    """
    # 将参数字典中的权重和偏置转换为张量
    W1 = tf.convert_to_tensor(parameters["W1"])  # 第一层权重转换为张量
    b1 = tf.convert_to_tensor(parameters["b1"])  # 第一层偏置转换为张量
    W2 = tf.convert_to_tensor(parameters["W2"])  # 第二层权重转换为张量
    b2 = tf.convert_to_tensor(parameters["b2"])  # 第二层偏置转换为张量
    W3 = tf.convert_to_tensor(parameters["W3"])  # 第三层权重转换为张量
    b3 = tf.convert_to_tensor(parameters["b3"])  # 第三层偏置转换为张量
    W4 = tf.convert_to_tensor(parameters["W4"])  # 第四层权重转换为张量
    b4 = tf.convert_to_tensor(parameters["b4"])  # 第四层偏置转换为张量
    
    # 创建一个新的参数字典，包含转换后的张量
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}
    
    # 创建一个浮点型占位符，用于输入数据
    x = tf.placeholder("float")
    
    # 使用前向传播函数计算模型输出
    z4 = forward_propagation_for_predict(x, params)
    
    # 获取具有最大值的索引作为预测类别
    p = tf.argmax(z4)
    
    # 创建TensorFlow会话
    sess = tf.Session()
    
    # 运行会话，计算预测结果，将输入数据X传入占位符x
    prediction = sess.run(p, feed_dict = {x: X})
    
    # 返回预测结果
    return prediction

# 预测概率函数
def predict_probability(X, parameters):
    # input -- X (dataset used to make prediction), papameters after training
    # output -- prediction probabilities
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}
    x = tf.placeholder("float")
    z4 = forward_propagation_for_predict(x, params)
    p = tf.nn.softmax(z4, axis=0)
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
    return prediction

# 神经网络模型训练函数
def model(X_train, Y_train, X_test, starting_learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 128, print_cost = True):
    # 输入 -- X_train (训练集), Y_train(训练标签), X_test (测试集)
    # 输出 -- 训练好的参数
    ops.reset_default_graph() # 重置TensorFlow计算图，清除之前定义的所有操作和变量
    tf.set_random_seed(3) # 设置TensorFlow随机种子，确保结果可重现
    seed = 3 # 初始化随机种子
    (nf, ns) = X_train.shape # 获取训练集的特征数量(nf)和样本数量(ns)
    nt = Y_train.shape[0] # 获取标签类别数量
    costs = [] # 初始化存储训练过程中代价的列表
    
    # 创建占位符，用于输入数据和标签
    X, Y = create_placeholders(nf, nt)
    # 初始化神经网络参数，定义网络结构：输入层nf个神经元，三个隐藏层分别有100,50,25个神经元，输出层nt个神经元
    parameters = initialize_parameters(nf=nf, ln1=100, ln2=50, ln3=25, nt=nt)
    # 构建前向传播网络，计算预测值
    Z4 = forward_propagation(X, parameters)
    # 计算代价函数，包含L2正则化(正则化系数为0.005)
    cost = compute_cost(Z4, Y, parameters, 0.005)
    
    # 使用学习率衰减策略
    global_step = tf.Variable(0, trainable=False) # 创建全局步数变量，不参与训练
    # 定义指数衰减学习率：每1000步学习率衰减为原来的0.95倍
    learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 1000, 0.95, staircase=True)
    # 使用Adam优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # 定义训练操作，最小化代价函数
    trainer = optimizer.minimize(cost, global_step=global_step)
    
    # 初始化所有TensorFlow变量
    init = tf.global_variables_initializer()
    
    # 开始训练过程
    with tf.Session() as sess:
        sess.run(init) # 执行变量初始化
        for epoch in range(num_epochs): # 迭代训练指定的轮数
            epoch_cost = 0. # 初始化当前轮的总代价
            num_minibatches = int(ns / minibatch_size) # 计算每轮需要的小批量数量
            seed = seed + 1 # 更新随机种子
            # 将训练数据分成多个小批量
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches: # 对每个小批量进行训练
                (minibatch_X, minibatch_Y) = minibatch # 获取当前小批量的数据和标签
                # 执行训练操作并计算当前小批量的代价
                _ , minibatch_cost = sess.run([trainer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                # 累加代价并计算平均值
                epoch_cost += minibatch_cost / num_minibatches
            # 每5轮打印一次训练进度
            if print_cost == True and (epoch+1) % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch+1, epoch_cost))
                costs.append(epoch_cost) # 记录代价历史
        
        # 获取训练后的参数值
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        # 计算训练集上的准确率
        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y)) # 检查预测类别是否与真实类别相同
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # 计算准确率
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train})) # 输出训练准确率
        return parameters # 返回训练好的参数

# 主程序
if __name__ == '__main__':
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()
    
    # 读取训练集数据文件
    train_set = pd.read_hdf(args.train_set, key="dge")
    # 将基因名称转换为大写
    train_set.index = [s.upper() for s in train_set.index]
    # 移除重复的基因名，保留第一次出现的
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]
    # 读取训练集标签文件
    train_label = pd.read_csv(args.train_label, header=None, sep="\t")
    # 读取测试集数据文件
    test_set = pd.read_hdf(args.test_set, key="dge")
    # 将测试集基因名称转换为大写
    test_set.index = [s.upper() for s in test_set.index]
    # 移除测试集中重复的基因名，保留第一次出现的
    test_set = test_set.loc[~test_set.index.duplicated(keep='first')]
    # 获取测试集的细胞条形码列表
    barcode = list(test_set.columns)
    
    # 计算训练集中不同细胞类型的数量
    nt = len(set(train_label.iloc[:,1]))
    
    # 对训练集和测试集进行数据标准化处理
    train_set, test_set = scale_sets([train_set, test_set])
    
    # 创建细胞类型到数字标签的映射字典
    type_to_label_dict = type_to_label_dict(train_label.iloc[:,1])
    # 创建数字标签到细胞类型的反向映射字典
    label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}
    # 打印训练集中的细胞类型
    print("Cell Types in training set:", type_to_label_dict)
    # 打印训练集中的细胞数量
    print("# Trainng cells:", train_label.shape[0])
    
    # 将训练标签从文本转换为数字标签
    train_label = convert_type_to_label(train_label.iloc[:,1], type_to_label_dict)
    # 将数字标签转换为独热编码形式
    train_label = one_hot_matrix(train_label, nt)
    
    # 训练神经网络模型，传入训练集、训练标签、测试集和训练参数
    parameters = model(train_set, train_label, test_set, \
    args.learning_rate, args.num_epochs, args.minibatch_size, args.print_cost)
    
    # 如果需要输出预测概率
    if args.output_probability:
        # 计算测试集中每个细胞属于各个类型的概率
        test_predict = pd.DataFrame(predict_probability(test_set, parameters))
        # 将行索引设置为细胞类型名称
        test_predict.index = [label_to_type_dict[x] for x in range(test_predict.shape[0])]
        # 将列名设置为细胞条形码
        test_predict.columns = barcode
        # 将预测概率保存到文件
        test_predict.to_csv("predicted_probabilities.txt", sep="\t")
    
    # 对测试集进行细胞类型预测
    test_predict = predict(test_set, parameters)
    # 将数字标签转换回细胞类型名称
    predicted_label = []
    for i in range(len(test_predict)):
        predicted_label.append(label_to_type_dict[test_predict[i]])
    # 创建包含细胞名称和预测类型的数据框
    predicted_label = pd.DataFrame({"cellname":barcode, "celltype":predicted_label})
    # 将预测结果保存到文件
    predicted_label.to_csv("predicted_label.txt", sep="\t", index=False)

# 计算并打印程序运行总时间
print("Run time:", timeit.default_timer() - run_time)
