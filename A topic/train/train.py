# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from enum import Enum
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE

class Is_Zombie(Enum):
    true = 1;
    false = 0;

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))


#load data
data = pd.read_csv('data_train_add.csv', header = 0)
# data = data.fillna('0')
print(data.isnull().any())
dataset = data.values
# print(dataset)
# smote = SMOTE(random_state=42)
# X = dataset[0:57438, [5,6,7,8,9,10,17,18,20]].astype(float)
# Y = dataset[0:57438, 21]
# encoder_Y = [0]*39798 + [1]*17640
X = dataset[0:57438, [5,6,7,8,9,10,17,18,20]].astype(float)
Y = dataset[0:57438, 21]
encoder_Y = [0]*39798 + [1]*17640
dummy_Y = np_utils.to_categorical(encoder_Y)
print(data.describe())

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)

# # 过采样前后的类别分布
# print(Y_train)
# print("过采样前训练集中类别0的数量：", sum(Y_train==0))
# print("过采样前训练集中类别1的数量：", sum(Y_train==1))
#
# # 进行过采样
# X_train, Y_train = smote.fit_resample(X_train, Y_train)
#
# print("过采样后训练集中类别0的数量：", sum(Y_train==0))
# print("过采样后训练集中类别1的数量：", sum(Y_train==1))
#
# print(Y_train)

# build keras model
model = Sequential()
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=4, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=2, activation='softmax'))  # units = nums of classes


# training
his = LossHistory()
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
model.summary()
model.save('weight.h5')
# his.loss_plot('epoch')

#
# # evaluate and draw confusion matrix
print('Test:')
score, accuracy = model.evaluate(X_test,Y_test,batch_size=32)
print('Test Score:{:.3}'.format(score))
print('Test accuracy:{:.3}'.format(accuracy))
# # confusion matrix
# Y_pred = model.predict(X_test)
# np.set_printoptions(precision=2)
# plt.figure()


