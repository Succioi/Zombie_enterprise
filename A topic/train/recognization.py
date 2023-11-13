import pandas as pd
import numpy as np
data = pd.read_csv('data_train.csv')
data = data.loc[:,('loss_cnt','Label')]
data = data.values
TP, TN = 0, 0
FP, FN = 0, 0
array = np.array(data)
print(array)
for i in range(array.shape[0]):
    if (array[i][0] >= 3) and (array[i][1] == 1):
        TP += 1
    if (array[i][0] >= 3) and (array[i][1] == 0):
        TN += 1
    if (array[i][0] < 3) and (array[i][1] == 1):
        FP += 1
    if (array[i][0] < 3) and (array[i][1] == 0):
        FN += 1
P=(TP/(TP+FP))
R=(TP/(TP+FN))
print('acc:',(TP+TN)/(TP+TN+FN+FP))
print('pre:',P)
print('rec:',R)
print('F1:',(2*P*R/(P+R)))