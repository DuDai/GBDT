import numpy as np
import pandas as pd

#树节点的类
class Node():
    def __init__(self, f_idx = -1, f_val= None, leaf_value= None, left= None, right= None):
        self.f_idx = f_idx
        self.f_val = f_val
        self.leaf_value = leaf_value
        self.left = left
        self.right = right

#计算叶子节点的数值
##为了便于实现，dataSet类型ndarray, dataSet[X, Y, y_res]
def leaf(dataSet):
     return np.sum(dataSet[:, -1]) / (np.sum((dataSet[:, -2] - dataSet[:, -1]) * (1- dataSet[:, -2] + dataSet[:, -1])))

#计算误差
def err_cnt(dataSet):

    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

#根据特征索引f_idx中的split_val将数据集划分为左右子树
def split_tree(dataSet, f_idx, f_val):
    set_L = dataSet[np.nonzero(dataSet[:, f_idx] <= f_val)[0], :]
    set_R = dataSet[np.nonzero(dataSet[:, f_idx] > f_val)[0], :]
    return set_L, set_R

#CART类
class CART_regression(object):
    def __init__(self, X, Y, min_sample, min_error, max_height= 20):
        self.X= X
        self.Y= Y
        self.min_sample = min_sample
        self.min_error= min_error
        self.max_height = max_height

    def fit(self):
        #将样本特征与样本标签整合成完整的样本
        data = np.c_[self.X, self.Y] #整合X, Y, residual
        #初始化
        best_err = err_cnt(data)
        #存储最佳分割属性及最佳切分点
        bestCriteria = None
        #存储切分后的两个数据集
        bestSets = None
        #构建决策树，返回该决策树的根节点
        ##若不满足min_sample, min_error, max_height等条件，则停止分割
        if np.shape(data)[0] <= self.min_sample or self.max_height == 1 or best_err <= self.min_error:
            return Node(leaf_value= leaf(data))

        #开始构建CART回归树
        num_feature = np.shape(data[0])[0] - 2
        for f_idx in range(num_feature):
            val_fea = np.unique(data[:, f_idx])
            for val in val_fea:
                #尝试划分
                set_L, set_R = split_tree(data, f_idx, val)
                if np.shape(set_L)[0] < 2 or np.shape(set_R)[0] < 2:
                    continue
                #计算划分后的error值
                err_now = err_cnt(set_L) + err_cnt(set_R)
                #更新最新划分
                if err_now < best_err:
                    best_err = err_now
                    bestCriteria= (f_idx, val)
                    bestSets= (set_L, set_R)
        #生成左右子树
        left = CART_regression(bestSets[0][:, :-1], bestSets[0][:, -1], self.min_sample, self.min_error, self.max_height-1).fit() #bestSets[0][:, : -1], bestSets[0][:, -1]在下一步会被整合，这里的顺序不乱就行
        right = CART_regression(bestSets[1][:, :-1], bestSets[1][:, -1], self.min_sample, self.min_error, self.max_height-1).fit()
        return Node(f_idx= bestCriteria[0], f_val=bestCriteria[1], left=left, right= right)

#CART预测
def predict(sample, tree):
    #如果tree是决策树桩
    if tree.leaf_value is not None:
        return tree.leaf_value
    else:
        if sample[tree.f_idx] <= tree.f_val:
            return predict(sample, tree.left)
        else:
            return predict(sample, tree.right)


