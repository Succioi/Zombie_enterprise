# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from enum import Enum

class Is_Zombie(Enum):
    true = 1;
    false = 0;

model = load_model('train/weight.h5')
data_test = pd.read_csv('n1.csv')#此处使用测试集
data_test = data_test.fillna("0")
data_test = data_test.loc[:,['registered_fund','employee_cnt','anche_year','assgro','liagro','vendinc','maibusinc','progro','netinc','ratgro','totequ','pat_cnt','mark_cnt','soft_cnt','works_cnt','ass_lia_rat','ROA','IP','loss_cnt']]
print(data_test)
X = data_test.iloc[:, :].astype(float)
transfer=MinMaxScaler(feature_range=(0,1))
X = transfer.fit_transform(X)
print(X)

# test_np = np.array(data_test)
# test_np = test_np.reshape(-1, 19)
test_np = np.array(X).reshape(-1, 19)
# print(test_np)
# print(test_np.size)
for size in range(np.size(test_np,0)):
    pred = np.argmax(model.predict(test_np))
    init_label = Is_Zombie(pred).name
    print(init_label)
