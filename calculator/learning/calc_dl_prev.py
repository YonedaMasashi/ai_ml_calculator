import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


plusData = pd.read_csv('plus.csv', sep=",", comment="#")

#...x(説明変数)、y(目的変数) への分割
x = plusData.values[:, 2:]
y = plusData.values[:, 1]

#...x, y の学習用・テスト用への分割
#......test_size : テストに使用するデータの割合
#......random_state : ランダムに分割する際の乱数のシード
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# 標準化
scaler = StandardScaler().fit(x_train)
x_train_transformed = scaler.transform(x_train)
x_test_transformed = scaler.transform(x_test)

# 全結合のニューラルネットワークに対応する MLPClassifierを読込
print("...MLPClassifier で学習")
classifier = MLPClassifier()
classifier.fit(x_train_transformed, y_train)
print(classifier.score(x_test_transformed, y_test))
print("")


print("データの保存と読込")
# 前処理用のプログラムを新たに作成し、前処理と保存を行う
np.savetxt("x_train_transformed.csv", x_train_transformed, delimiter=",")
np.savetxt("x_test_transformed.csv", x_test_transformed, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
np.savetxt("y_test.csv", y_test, delimiter=",")

