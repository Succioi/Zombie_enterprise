# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from enum import Enum
import matplotlib as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.models import Sequential
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


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

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

#load data
data = pd.read_csv('data_train.csv', header = 0)
dataset = data.values
X = dataset[0:40778, 2:21].astype(float)  # 忽略run数据
Y = dataset[0:40778, 21]
encoder_Y = [0]*39798 + [1]*980
dummy_Y = np_utils.to_categorical(encoder_Y)


# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)

# build keras model
model = Sequential()
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
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


