import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("data_train.csv")
data1 = pd.read_csv("data_trian_light.csv")
data_train1 = data.fillna('0')
data_test1 = data1.fillna('0')
np.random.seed(123)

# 建立逻辑回归模型
pre_x = LogisticRegression()
pre_x.fit(data_train1[['employee_cnt','assgro','liagro','vendinc','maibusinc','progro','netinc','ass_lia_rat','ROA','loss_cnt']], data_train1['Label'])

# 预测训练集
predict_x = pre_x.predict_proba(data_train1[['employee_cnt','assgro','liagro','vendinc','maibusinc','progro','netinc','ass_lia_rat','ROA','loss_cnt']])[:, 1]
pd.crosstab(predict_x, data['Label'])
fpr_x1, tpr_x1, thresholds_x1 = roc_curve(data_train1['Label'], predict_x)
# 计算 AUC
auc_x1 = roc_auc_score(data_train1['Label'], predict_x)
# 绘制 ROC 曲线
plt.plot(fpr_x1, tpr_x1, label='AUC = {:.4f}'.format(auc_x1))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Training Set')
plt.legend(loc='lower right')
plt.show()

# 预测测试集
predict_x1 = pre_x.predict_proba(data_test1[['employee_cnt','assgro','liagro','vendinc','maibusinc','progro','netinc','ass_lia_rat','ROA','loss_cnt']])[:, 1]
fpr_x2, tpr_x2, thresholds_x2 = roc_curve(data_test1['Label'], predict_x1)
# 计算 AUC
auc_x2 = roc_auc_score(data_test1['Label'], predict_x1)
# 绘制 ROC 曲线
plt.plot(fpr_x2, tpr_x2, label='AUC = {:.4f}'.format(auc_x2))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Test Set')
plt.legend(loc='lower right')
plt.show()

# 计算模型精度
predict1 = pre_x.predict(data_test1[['employee_cnt','assgro','liagro','vendinc','maibusinc','progro','netinc','ass_lia_rat','ROA','loss_cnt']])
accuracy = accuracy_score(data_test1['Label'], predict1)
# 计算混淆矩阵
cm = confusion_matrix(data_test1['Label'], predict1)
# 计算精确率
precision = precision_score(data_test1['Label'], predict1)
# 计算召回率
recall = recall_score(data_test1['Label'], predict1)
# 计算 F-Measure
F_measure = f1_score(data_test1['Label'], predict1)
# 输出以上各结果
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F-Measure:', F_measure)
print('Confusion Matrix:\n', cm)
