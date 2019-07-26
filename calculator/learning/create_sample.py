import pandas as pd
from numpy.random import *
import csv
import sys

args = sys.argv

print(args)
print("第1引数(最大値)：" + args[1])
print("第2引数(個数)：" + args[2])

max_int = int(args[1])
number = int(args[2])

#  0〜4999 の整数を5000個生成
firsts = randint(0,max_int,number)
seconds = randint(0,max_int,number)

values = list(zip(firsts, seconds))
answers = []
for key, value in values:
    answers.append(key + value)

datas = zip(answers, firsts, seconds)
csvFile = pd.DataFrame(datas, columns=['answer', 'first', 'second'])
csvFile.to_csv('plus.csv')

