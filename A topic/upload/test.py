# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

data_true = pd.read_csv("data_test_add_true.csv")
data_test = pd.read_csv("data_test_add.csv")
# 查看数据的基本信息
print(data_test.info())
print(data_test.head())

# 载入权重模型
model = load_model("weight/weight2_data_train_19_SMOTE.h5")
# 定义自变量
X_new = data_test.drop(["ent_id","ent_type"], axis=1)
# X_new = data_test.loc[:,['assgro','liagro','vendinc','maibusinc','progro','netinc','ass_lia_rat','ROA','loss_cnt']]

# 归一化数据
scaler = MinMaxScaler()
X_new = scaler.fit_transform(X_new)

# 预测新的数据
y_new = model.predict(X_new)
print(y_new)
print("***************************************************************")
print("***************************************************************")
print("***************************************************************")
y_new = (y_new > 0.48).astype(int)
print(y_new)
# 19_SMOTE:0.46 ~ 0.5 ; 19_ADASYN:0.2 ; 19_ClusterCentroids:0.9(这权重效果不太好) ;
#  9_SMOTE:0.4 ;  9_ADASYN:0.4 ;  9_ClusterCentroids:0.5

# 将预测结果添加到新的数据中
data_test["zombie_pred"] = y_new
# 保存新的数据到一个文件中
data_test.to_csv("new_data_pred.csv", index=False)

print("Test:")
score, accuracy = model.evaluate(X_new,y_new,batch_size=64)
f1 = f1_score(y_true=data_true,y_pred=y_new)
print("Test Score:{:.3}".format(score))
print("Test accuracy:{:.3}".format(accuracy))
print("f1_score:{:.3}".format(f1))