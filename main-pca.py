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
        pca_r = int(round(retain_variance, 2) * 100)
        joblib.dump(pca, f"./model/pca{pca_r}.pkl")
        return data

    def train(self):
        data = pd.read_csv(self.train_path)
        x = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        # 特征值归一化
        x = x / 255

        # 对图片进行pca降维
        for retain_variance in np.arange(0.3, 1.0, 0.05):
            pca_x = self.data_reduction(x,retain_variance)

            x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=0.2)

            # 定义不同的核函数
            #Kernel = ["linear", "poly", "rbf", "sigmoid"]
            Kernel = ["rbf"]
            scores = []

            for kernel in Kernel:
                svc = svm.SVC(kernel=kernel)
                svc.fit(x_train, y_train)
                score = svc.score(x_test, y_test)
                scores.append(score)
                pca_r=int(round(retain_variance,2)*100)
                joblib.dump(svc, f"./model/mnist_svc_{kernel}_{pca_r}.pkl")
                print(f"Kernel: {kernel}, pca: {retain_variance},Score: {score}")

        return Kernel, scores


if __name__ == '__main__':
    mn = Mnist("./mnist_train.csv", './mnist_test.csv')
    Kernel, scores = mn.train()

