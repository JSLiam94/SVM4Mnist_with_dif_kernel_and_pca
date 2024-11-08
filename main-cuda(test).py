import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from thundersvm import SVC
from sklearn.preprocessing import StandardScaler

class Mnist(object):
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def data_reduction(self, data):
        pca = PCA(0.85)
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
        pca_x = self.data_reduction(x)

        x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=0.2)

        # 定义不同的核函数
        Kernel = ["linear", "poly", "rbf", "sigmoid"]
        scores = []

        for kernel in Kernel:
            svc = SVC(kernel=kernel, max_iter=96000, verbose=True)  # 使用thundersvm的SVC
            svc.fit(x_train, y_train)
            score = svc.score(x_test, y_test)
            scores.append(score)
            joblib.dump(svc, f"./model/mnist_svc_{kernel}.pkl")
            print(f"Kernel: {kernel}, Score: {score}")


        return Kernel, scores

    def plot_decision_boundaries(self, X, y, Kernel):
        nrows = len(Kernel)
        ncols = 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, nrows * 4))

        for ax_idx, kernel in enumerate(Kernel):
            ax = axes[ax_idx]
            clf = SVC(kernel=kernel).fit(X, y)
            X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.1),
                                np.arange(Y_min, Y_max, 0.1))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4)
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
            ax.set_title(f"{kernel} kernel")
            ax.set_xticks(())
            ax.set_yticks(())

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    mn = Mnist("./mnist_train.csv", './mnist_test.csv')
    Kernel, scores = mn.train()
    data = pd.read_csv(mn.train_path)
    x = data.iloc[:, 1:].values / 255.0
    y = data.iloc[:, 0].values
    pca = joblib.load("./model/pca.pkl")
    x_pca = pca.fit_transform(x)
    mn.plot_decision_boundaries(x_pca, y, Kernel)