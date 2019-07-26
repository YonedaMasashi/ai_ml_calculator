import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.initializers import TruncatedNormal

# データの保存と読込
x_train_transformed = np.loadtxt("x_train_transformed.csv", delimiter=",")
x_test_transformed = np.loadtxt("x_test_transformed.csv", delimiter=",")
y_train = np.loadtxt("y_train.csv", delimiter=",")
y_test = np.loadtxt("y_test.csv", delimiter=",")
print(y_test)
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


# Kerasによる学習とモデルの評価
print("Kerasによる学習とモデルの評価")
model.fit(x_train_transformed, y_train, epochs=200, batch_size=64, validation_split=0.2)
print("")

# TensorBoardによる精度とLossの可視化
print("TensorBoardによる精度とLossの可視化")
tb_cd = TensorBoard(log_dir="tb_log/")
model.fit(x_train_transformed, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[tb_cd])
print("")

# テストデータでの評価と推論
print("テストデータでの評価と推論")
score = model.evaluate(x_test_transformed, y_test)
print("loss:", score[0])
print("accuracy:", score[1])
print(model.predict(x_test_transformed[0:10,:]))
print("")

# 任意の学習済モデルを用いた評価と推論
# ...学習途中で随時モデルを保存する
print("任意の学習済モデルを用いた評価と推論")
fpath = './model/weights.{epoch:03d}-{loss:.2f}.hdf5' # モデルの保存先
cp_cd = ModelCheckpoint(filepath = fpath, period=5)   # 定期的にモデルを保存。5エポック毎に保存

model.fit(x_train_transformed, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[tb_cd, cp_cd])
