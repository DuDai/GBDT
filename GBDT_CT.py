import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import CART_regression_tree

#GBDT_RT类
class GBDT_CT(object):
    def __init__(self, n_estimater= 10, learn_rate= 0.5):
        self.n_estimater = n_estimater
        self.learn_rate = learn_rate
        self.init_value = None
        self.trees = []

    #计算预测标签和残差
    def get_init_value(self, Y):
        p = np.count_nonzero(Y) #计算类别1的样本个数
        n = np.shape(Y)[0] #总样本个数
        return np.log(p / (n-p))

    #计算sigmoid函数
    def get_sigmoid(self, y):
        if y >= 0:
            return 1/ (1 + np.exp(- y))
        else:
            return np.exp(y) / (1 + np.exp(y))

    def fit(self, X, Y, min_sample, min_error, max_height):

        n = np.shape(Y)[0] #样本个数
        #初始化预测标签和残差
        self.init_value = self.get_init_value(Y)
        F = np.array([self.init_value] * n) #初始化第一个弱学习器
        y_hat = np.array([self.get_sigmoid(self.init_value)] * n) #第一个模型的预测值
        y_residuals = Y - y_hat #第一个模型的残差
        #为了后续便于计算残差的拟合值来近似残差并叶子节点的值，这里把[Y，y_residuals]进行合并
        y_residuals = np.c_[Y, y_residuals]

        #迭代n_estimater次生成GBDT
        for j in range(self.n_estimater):
            #生成第 j 棵树
            tree = CART_regression_tree.CART_regression(X, y_residuals, min_sample, min_error, max_height).fit()
            #计算当前的模型预测值
            for k in range(n):
                res_hat = CART_regression_tree.predict(X[k], tree) #当前这棵树预测的残差
                #计算此时的模型预测值=原预测值+残差预测值
                F[k] += self.learn_rate * res_hat
                y_hat[k] = self.get_sigmoid(F[k])
            #计算当前模型的残差
            y_residuals = Y - y_hat
            #整合
            y_residuals = np.c_[Y, y_residuals]
            self.trees.append(tree)

    #模型预测
    def GBDT_predict(self, X_test):
        predicts = []
        for i in range(np.shape(X_test)[0]):
            pre_y = self.init_value
            for tree in self.trees:
                pre_y += self.learn_rate * CART_regression_tree.predict(X_test[i], tree)
            if self.get_sigmoid(pre_y) >= 0.5:
                predicts.append(1)
            else:
                predicts.append(0)
        return predicts

    #计算误差
    def cal_error(self, Y_test, predicts):
        y_test = np.array(Y_test)
        y_predicts = np.array(predicts)
        error = np.square(y_test - y_predicts).sum() / len(Y_test)
        return error

if __name__ == "__main__":
    #加载数据集
    cwd = os.getcwd()
    datasets_path = os.path.join(cwd, 'data_final_2.xlsx')
    df = pd.read_excel(datasets_path)
    #设置随机种子
    # np.random.seed(123)
    #划分训练集、测试集
    # X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values, test_size= 0.2, random_state= 123)
    Train, val_test = train_test_split(df, test_size=0.4, random_state=0, shuffle=True)  # 训练集占60%
    val, test = train_test_split(df, test_size=0.5, random_state=0, shuffle=True)  # 验证集和测试集各占20%

    X_Train = Train.iloc[:, :14].values
    Y_Train = Train.iloc[:, 14].values

    X_Val = val.iloc[:, :14].values
    Y_Val = val.iloc[:, 14].values

    X_Test = test.iloc[:, :14].values
    Y_Test = test.iloc[:, 14].values

    #设置模型参数
    n_estimater = 50
    learn_rate = 0.2
    min_sample = 30
    min_error = 0.3
    max_height = 8  # 深度为属性值的一半

    #实例化GBDT_RT
    gbdtTrees = GBDT_CT(n_estimater=n_estimater, learn_rate=learn_rate)

    #拟合模型
    gbdtTrees.fit(X_Train, Y_Train, min_sample, min_error, max_height)

    #模型预测
    Y_Pred = gbdtTrees.GBDT_predict(X_Val)
    Y_Pred2 = gbdtTrees.GBDT_predict(X_Test)
    print("Y_Pred=", Y_Pred)
    print("Y_Pred2=", Y_Pred2)

    # 混淆矩阵
    cm = confusion_matrix(Y_Val, Y_Pred)
    cm2 = confusion_matrix(Y_Test, Y_Pred2)
    print('confusion matrix:')
    print(cm)
    print('confusion matrix2:')
    print(cm2)

    #计算验证集和测试集准确率
    acc = accuracy_score(Y_Val, Y_Pred)
    print("acc=", acc)
    acc2 = accuracy_score(Y_Test, Y_Pred2)
    print("acc2=", acc2)

