import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import sklearn
import pandas as pd
from sklearn.datasets import load_iris

# 直接从sklearn中获取数据集
iris = datasets.load_iris()
X = iris.data[:, :4]    # 表示我们取特征空间中的4个维度
print(X.shape)

iris_dataset = load_iris()
#调用 load_iris 函数来加载数据
#load_iris 返回的 iris 对象是一个 Bunch 对象，与字典非常相似，里面包含键和值

#了解数据
print("keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset)
print(iris_dataset['DESCR']+"\n..")

print("target names: \n{}".format(iris_dataset['data']))

print("target names: \n{}".format(iris_dataset['target_names']))

print("Feature names: \n{}".format(iris_dataset['feature_names']))
#'sepal length':花萼长度 'sepal width':花萼宽度 'petal length':花瓣长度 'petal width':花瓣宽度

# 法一：直接手写实现
# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 为给定数据集构建一个包含K个随机质心centroids的集合
def randCent(dataSet, k):
    m, n = dataSet.shape  # m=150,n=4
    centroids = np.zeros((k, n))  # 4*4
    for i in range(k):  # 执行四次
        index = int(np.random.uniform(0, m))  # 产生0到150的随机数（在数据集中随机挑一个向量做为质心的初值）
        centroids[i, :] = dataSet[index, :]  # 把对应行的四个维度传给质心的集合
    return centroids


# k均值聚类算法
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行数150
    # 第一列存每个样本属于哪一簇(四个簇)
    # 第二列存每个样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))  # .mat()创建150*2的矩阵
    clusterChange = True
    # 1.初始化质心centroids
    centroids = randCent(dataSet, k)  # 4*4
    while clusterChange:
        # 样本所属簇不再更新时停止迭代
        clusterChange = False
        # 遍历所有的样本（行数150）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有的质心
            # 2.找出最近的质心
            for j in range(k):
                # 计算该样本到4个质心的欧式距离，找到距离最近的那个质心minIndex
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 3.更新该行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 4.更新质心
        for j in range(k):
            # np.nonzero(x)返回值不为零的元素的下标，它的返回值是一个长度为x.ndim(x的轴数)的元组
            # 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。
            # 矩阵名.A 代表将 矩阵转化为array数组类型

            # 这里取矩阵clusterAssment所有行的第一列，转为一个array数组，与j（簇类标签值）比较，返回true or false
            # 通过np.nonzero产生一个array，其中是对应簇类所有的点的下标值（x个）
            # 再用这些下标值求出dataSet数据集中的对应行，保存为pointsInCluster（x*4）
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取对应簇类所有的点（x*4）
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 求均值，产生新的质心
            # axis=0，那么输出是1行4列，求的是pointsInCluster每一列的平均值，即axis是几，那就表明哪一维度被压缩成1
    print("cluster complete")
    return centroids, clusterAssment


def draw(data, center, assment):
    length = len(center)
    fig = plt.figure
    data1 = data[np.nonzero(assment[:, 0].A == 0)[0]]
    data2 = data[np.nonzero(assment[:, 0].A == 1)[0]]
    data3 = data[np.nonzero(assment[:, 0].A == 2)[0]]
    # 选取前两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 0], data1[:, 1], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 0], data2[:, 1], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 0], data3[:, 1], c="blue", marker='+', label='label2')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 0], center[i, 1]), xytext= \
            (center[i, 0] + 1, center[i, 1] + 1), arrowprops=dict(facecolor='yellow'))
        #  plt.annotate('center',xy=(center[i,0],center[i,1]),xytext=\
        # (center[i,0]+1,center[i,1]+1),arrowprops=dict(facecolor='red'))
    plt.show()
    # 选取后两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 2], data1[:, 3], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 2], data2[:, 3], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 2], data3[:, 3], c="blue", marker='+', label='label2')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 2], center[i, 3]), xytext= \
            (center[i, 2] + 1, center[i, 3] + 1), arrowprops=dict(facecolor='yellow'))
    plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 iris_dataset['data'], iris_dataset['target'], random_state=0)
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
'''scikit-learn 中的 train_test_split 函数可以打乱数据集并进行拆分。这个函数将 75% 的
行数据及对应标签作为训练集，剩下 25% 的数据及其标签作为测试集。训练集与测试集的分配比例
可以是随意的，但使用 25% 的数据作为测试集是很好的经验法则。利用 random_state 参数指定了
随机数生成器的种子。这样函数输出就是固定不变的，所以这行代码的输出始终相同'''

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
  hist_kwds={'bins': 20}, s=60, alpha=.8)  #画出散点图

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) #k近邻分类器
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

knn = KNeighborsClassifier(n_neighbors=25)
knn = KNeighborsClassifier(n_neighbors=50)

dataSet = X
k = 3
centroids,clusterAssment = KMeans(dataSet,k)
draw(dataSet,centroids,clusterAssment)

# 取前两个维度（萼片长度、萼片宽度），绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

def Model(n_clusters):
    estimator = KMeans(n_clusters=n_clusters)# 构造聚类器
    return estimator

def train(estimator):
    estimator.fit(X)  # 聚类

# 初始化实例，并开启训练拟合
estimator=Model(3)
train(estimator)

label_pred = estimator.labels_  # 获取聚类标签
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()
