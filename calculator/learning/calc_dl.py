import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.initializers import TruncatedNormal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

def learningDL():
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

    # Kerasによるニューラルネットワークの構築
    print("Kerasによるニューラルネットワークの構築")
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=2))
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1,  kernel_initializer=TruncatedNormal(stddev=0.01)))
    print("ネットワークの重みの表示:", model.get_weights())
    print("")

    # Kerasによる学習
    print("Kerasによる学習")
    # 損失関数の設定
    model.compile(loss='binary_crossentropy', # 2値の分類問題の場合、binary_crossentropy を使用
        optimizer='sgd',                      # 最適化アルゴリズムの指定
        metrics=['accuracy'])                 # 評価関数を指定。これを書くと精度が表示される
    # 学習を実行
    model.fit(x_train_transformed, y_train,   # 入力するデータと、教師データ
        epochs=200,                           # エポック数
        batch_size=64)                        # バッチサイズ
    print("")

    with open("KerasLearning.pickle", mode='wb') as fp:
        pickle.dump(model, fp)
