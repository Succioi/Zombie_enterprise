# 导入所需的库
import pandas as pd
from keras import layers
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import ClusterCentroids

data = pd.read_csv("data_train.csv")
data = data.fillna('0')
# 查看数据的基本信息
print(data.info())
print(data.head())

# 定义自变量和因变量
X = data.drop(['ent_id','ent_type','Label'], axis=1)
# X = data.loc[:,['assgro','liagro','vendinc','maibusinc','progro','netinc','ass_lia_rat','ROA','loss_cnt']]

y = data["Label"]

# adasyn = ADASYN()
# X, y = adasyn.fit_resample(X, y)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
# cc = ClusterCentroids(random_state=0)
# X, y = cc.fit_sample(X, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 归一化数据
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 建立深度神经网络模型
model = Sequential([
    layers.Dense(64, activation="relu", input_shape=[X_train.shape[1]]),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

# 编译模型
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 训练模型
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

# 保存权重
model.save('weight2_data_train_19_SMOTE.h5')

# 评估模型
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred))
