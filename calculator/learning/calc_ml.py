import pandas as pd
from numpy.random import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
import csv
import pickle

def learningML():
    plusData = pd.read_csv('plus.csv', sep=",", comment="#")

    #...x(説明変数)、y(目的変数) への分割
    x = plusData.values[:, 2:]
    y = plusData.values[:, 1]

    #...x, y の学習用・テスト用への分割
    #......test_size : テストに使用するデータの割合
    #......random_state : ランダムに分割する際の乱数のシード
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

    #...決定木
    print("決定木 ---------------------------------------------------")
    classifier = DecisionTreeClassifier()
    #...識別器の学習
    classifier.fit(x_train, y_train)

    with open("DecisionTreeClass.pickle", mode='wb') as fp:
        pickle.dump(classifier, fp)

    """
    #...学習済みのモデルを使った推論
    print("...学習済みのモデルを使った推論")
    print(classifier.predict(x_test))
    print("")
    #...正答率の計算
    print("...正答率の計算")
    print(classifier.score(x_test, y_test))
    print("")
    """


    """
    #...サポートベクターマシン
    print("サポートベクターマシン ---------------------------------------------------")
    clf = SVC()
    clf.fit(x_train, y_train)
    print("...SVC 学習済みのモデルを使った推論")
    print(clf.predict(x_test))
    print("")
    print("...SVC 正答率の計算")
    print(clf.score(x_test, y_test))
    print("")


    #...Cross-validation
    print("Cross-validation ---------------------------------------------------")
    #...arg1:識別器, arg2:予測に用いる値, arg3:予測したい値, arg4:分割する数
    print("...決定木")
    cv_score = cross_val_score(classifier, x, y, cv=5)
    print(cv_score) #...それぞれの validation のスコアを表示
    print(cv_score.mean())
    print("")

    print("...SVC")
    cv_score_svc = cross_val_score(clf, x, y, cv=5)
    print(cv_score_svc) #...それぞれの validation のスコアを表示
    print(cv_score_svc.mean())
    print("")

    #...データの標準化
    print("...データの標準化")
    #...渡されたデータの平均や標準偏差等を計算し内部的に保存
    scaler = StandardScaler().fit(x_train)
    x_train_transformed = scaler.transform(x_train)
    x_test_transformed = scaler.transform(x_test)
    classifier.fit(x_train_transformed, y_train)
    print(classifier.score(x_test_transformed, y_test))
    print("")

    #...ランダムフォレスト
    print("ランダムフォレスト ---------------------------------------------------")
    rndm_forest = RandomForestClassifier(random_state=0)
    rndm_forest = rndm_forest.fit(x_train, y_train)
    pred = rndm_forest.predict(x_train)
    fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)
    auc(fpr, tpr)
    print(accuracy_score(pred, y_test))
    print("")
    """

def predict(first, second):
    with open("DecisionTreeClass.pickle", mode='rb') as fp:
        model = pickle.load(fp)
        predected_answer = model.predict([[first, second]])
        return predected_answer
