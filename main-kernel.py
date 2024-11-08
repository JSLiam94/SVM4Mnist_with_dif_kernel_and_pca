import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.preprocessing import StandardScaler

class Mnist(object):
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def data_reduction(self, data,retain_variance):
        pca = PCA(retain_variance)
        data = pca.fit_transform(data)
        joblib.dump(pca, "./model/pca.pkl")
        return data

    def train(self):
        data = pd.read_csv(self.train_path)
        x = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        # 特征值归一化
        x = x / 255

        # 对图片进行pca降维
        retain_variance = 0.85 #降维比例
        
        pca_x = self.data_reduction(x,retain_variance)

        x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=0.2)

        # 定义不同的核函数
        Kernel = ["linear", "poly", "rbf", "sigmoid"]
            
        scores = []

        for kernel in Kernel:
                svc = svm.SVC(kernel=kernel)
                svc.fit(x_train, y_train)
                score = svc.score(x_test, y_test)
                scores.append(score)
                joblib.dump(svc, f"./model/mnist_svc_{kernel}.pkl")
                print(f"Kernel: {kernel}, pca: {retain_variance},Score: {score}")

        return Kernel, scores



    def plot_decision_boundaries(self,X, y, Kernel):
        for kernel in Kernel:
            plt.figure(figsize=(8, 6))  # 创建新的图形窗口
            clf = joblib.load(f"./model/mnist_svc_{kernel}.pkl")

            # 创建一个与PCA降维后的数据维度相匹配的网格
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))

            # 将二维网格扩展到与PCA降维后的数据相同的维度
            grid = np.c_[xx.ravel(), yy.ravel()]
            for i in range(2, X.shape[1]):
                grid = np.c_[grid, np.full(grid.shape[0], X[:, i].mean())]

            # 预测网格上的分类
            Z = clf.predict(grid)
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
            plt.title(f"{kernel} kernel")
            plt.xticks(())
            plt.yticks(())

            # 计算评估指标
            y_pred = clf.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='macro')
            recall = recall_score(y, y_pred, average='macro')
            f1 = f1_score(y, y_pred, average='macro')

            # 打印评估结果
            print(f"Kernel: {kernel}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 30)

            plt.show()  # 显示当前图形窗口

    def show_bound(self):
        data = pd.read_csv(mn.test_path)
        x = data.iloc[:, 1:].values / 255.0
        y = data.iloc[:, 0].values
        pca = joblib.load("./model/pca.pkl")
        Kernel = ["linear", "poly", "rbf", "sigmoid"]
        x_pca = pca.transform(x)  # 使用transform降维
        mn.plot_decision_boundaries(x_pca, y, Kernel)  # 使用PCA降维后的数据

if __name__ == '__main__':
    mn = Mnist("./mnist_train.csv", './mnist_test.csv')
    Kernel, scores = mn.train()
    mn.show_bound()
