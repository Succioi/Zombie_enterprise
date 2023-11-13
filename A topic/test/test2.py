# 导入所需的库
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
data_test = pd.read_csv("../n1.csv")
# 查看数据的基本信息
print(data_test.info())
print(data_test.head())

# 载入权重模型
model = load_model("C:/Users/11031/Desktop/A topic/train/weight2_data_train_9.h5")
# 定义自变量
# X_new = data_test.drop(["ent_id","ent_type"], axis=1)
X_new = data_test.loc[:,['assgro','liagro','vendinc','maibusinc','progro','netinc','ass_lia_rat','ROA','loss_cnt']]
# 归一化数据
scaler = MinMaxScaler()
X_new = scaler.fit_transform(X_new)

# 预测新的数据
y_new = model.predict(X_new)
y_new = (y_new > 0.5).astype(int)

# # 将预测结果添加到新的数据中
# data_test["zombie_pred"] = y_new
# # 查看新的数据的前几行
# print(data_test.head())
# # 保存新的数据到一个文件中
# data_test.to_csv("new_data_pred.csv", index=False)

print('Test:')
score, accuracy = model.evaluate(X_new,y_new,batch_size=64)
print('Test Score:{:.3}'.format(score))
print('Test accuracy:{:.3}'.format(accuracy))