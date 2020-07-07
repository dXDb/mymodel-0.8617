# -*- coding: utf-8 -*-
"""
Created on Fri May  8 03:23:59 2020

@author: CVPR
"""

import copy
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import initializers
from keras.datasets import cifar10

random.seed(123)
tf.set_random_seed(123)
np.random.seed(123)

# load data
(xx_train, yy_train), (xx_test, yy_test) = cifar10.load_data()
img_row = xx_train.shape[1]
img_col = xx_train.shape[2]
img_channel = xx_train.shape[3]

# make valid set
y_train = copy.deepcopy(yy_train[0:50000])

# init
t_mean = [0,0,0]
t_std = [0,0,0]
newX_train = np.ones(xx_train.shape)
newX_test = np.ones(xx_test.shape)

# (image - mean) / std => normalization
for i in range(3):
    t_mean[i] = np.mean(xx_train[:,:,:,i])
    t_std[i] = np.mean(xx_train[:,:,:,i])
    

    newX_train[:,:,:,i] = copy.deepcopy(xx_train[:,:,:,i]) - t_mean[i]
    newX_train[:,:,:,i] = newX_train[:,:,:,i] / t_std[i]
    newX_test[:,:,:,i] = xx_test[:,:,:,i] - t_mean[i]
    newX_test[:,:,:,i] = newX_test[:,:,:,i] / t_std[i]

X_train = newX_train
X_test = newX_test

# Hyper parameters
batchSize = 500            #-- Training Batch Size
num_classes = 10           #-- Number of classes in CIFAR-10 dataset
num_epochs = 5000            #-- Number of epochs for training   
learningRate= 0.05        #-- Learning rate for the network
lr_weight_decay = 0.02     #-- Learning weight decay. Reduce the learn rate by 0.95 after epoch
drop_out = 0.5
sec_drop_out = 0.3

# ---------------------------- model (VGG16) ----------------------------------
loss = []
acc = []


# -- one hot --
samples = len(y_train)
out = np.zeros((samples, 10))
for i in range(y_train.shape[0]):
    out[i][y_train[i][0]] = 1
y_train = out

samples = len(yy_test)
out = np.zeros((samples, 10))
for i in range(yy_test.shape[0]):
    out[i][yy_test[i][0]] = 1
y_test = out

# -- input layer --
x = tf.placeholder(tf.float32, shape=[None, img_row, img_col, img_channel])
y = tf.placeholder(tf.float32, shape=[None, 10])

fw = tf.keras.initializers.truncated_normal()
fb = tf.keras.initializers.Zeros()

# -- first --
w1_1 = tf.Variable(fw(shape=[3, 3, 3, 64]))
b1_1 = tf.Variable(fb(shape=[64]))
cnn1 = tf.nn.conv2d(x, w1_1, strides=[1, 1, 1, 1], padding='SAME') + b1_1
cnn1 = tf.nn.relu(cnn1)
        
w1_1 = tf.Variable(fw(shape=[3, 3, 64, 64]))
b1_1 = tf.Variable(fb(shape=[64]))
cnn1 = tf.nn.conv2d(cnn1, w1_1, strides=[1, 1, 1, 1], padding='SAME') + b1_1
cnn1 = tf.nn.relu(cnn1)

cnn1 = tf.nn.max_pool(cnn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

# -- second --
w2_1 = tf.Variable(fw(shape=[3, 3, 64, 128]))
b2_1 = tf.Variable(fb(shape=[128]))
cnn2 = tf.nn.conv2d(cnn1, w2_1, strides=(1, 1), padding='SAME') + b2_1
cnn2 = tf.nn.relu(cnn2)

w2_1 = tf.Variable(fw(shape=[3, 3, 128, 128]))
b2_1 = tf.Variable(fb(shape=[128]))
cnn2 = tf.nn.conv2d(cnn2, w2_1, strides=(1, 1), padding='SAME') + b2_1
cnn2 = tf.nn.relu(cnn2)

cnn2 = tf.nn.dropout(cnn2, drop_out)

w2_1 = tf.Variable(fw(shape=[3, 3, 128, 128]))
b2_1 = tf.Variable(fb(shape=[128]))
cnn2 = tf.nn.conv2d(cnn2, w2_1, strides=(1, 1), padding='SAME') + b2_1
cnn2 = tf.nn.relu(cnn2)

cnn2 = tf.nn.max_pool(cnn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

# -- second --
w3_1 = tf.Variable(fw(shape=[3, 3, 128, 256]))
b3_1 = tf.Variable(fb(shape=[256]))
cnn3 = tf.nn.conv2d(cnn2, w3_1, strides=(1, 1), padding='SAME') + b3_1
cnn3 = tf.nn.relu(cnn3)

w3_1 = tf.Variable(fw(shape=[3, 3, 256, 256]))
b3_1 = tf.Variable(fb(shape=[256]))
cnn3 = tf.nn.conv2d(cnn3, w3_1, strides=(1, 1), padding='SAME') + b3_1
cnn3 = tf.nn.relu(cnn3)

cnn3 = tf.nn.dropout(cnn3, drop_out)

w3_1 = tf.Variable(fw(shape=[3, 3, 256, 256]))
b3_1 = tf.Variable(fb(shape=[256]))
cnn3 = tf.nn.conv2d(cnn3, w3_1, strides=(1, 1), padding='SAME') + b3_1
cnn3 = tf.nn.relu(cnn3)

cnn3 = tf.nn.max_pool(cnn3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

# -- fifth --
W_fc1 = tf.Variable(fw(shape=[4 * 4 * 256, 512]))
h_conv5_flat = tf.layers.flatten(cnn3)
fc1_1 = tf.Variable(fb(shape=[512]))
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + fc1_1)

h_fc1 = tf.nn.dropout(h_fc1, drop_out)

W_fc1 = tf.Variable(fw(shape=[512, 256]))
fc1_1 = tf.Variable(fb(shape=[256]))
h_fc1 = tf.nn.relu(tf.matmul(h_fc1, W_fc1) + fc1_1)

# -- sevneth --
W_fc3 = tf.Variable(fw(shape=[256, 10]))
fc3_1 = tf.Variable(fb(shape=[10]))
logits = tf.matmul(h_fc1, W_fc3) + fc3_1
y_pred = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
l_r = tf.Variable(learningRate, dtype=tf.float32, shape=())
optimizer = tf.train.MomentumOptimizer(learning_rate=l_r, momentum=0.9).minimize(cost)


correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# init
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


print('Learning started. It takes sometimes.')
loss_list = list()
accuracy_list = list()
test_loss_list = list()
test_accuracy_list = list()
max_acc = 0

for epoch in range(num_epochs):

    loss_batch = 0
    test_loss_batch = 0
    accuracy_batch = 0
    test_accuracy_batch = 0
    total_batch = X_train.shape[0] // batchSize

    # --train--
    x_train_batch = X_train.reshape(-1, batchSize, 32, 32, 3)
    y_train_batch = y_train.reshape(-1, batchSize, 10)

    if epoch == 150: learningRate = sess.run(tf.compat.v1.assign(l_r, learningRate * lr_weight_decay))
    if epoch == 400: learningRate = sess.run(tf.compat.v1.assign(l_r, learningRate * lr_weight_decay))
    thislearningRate = learningRate

    for i in range(total_batch):
        feed_dict = {x: x_train_batch[i], y: y_train_batch[i]}
        c, accu, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)
        if i % 10 == 0: print("step:","%04d"%(i + 1), "cost =","{:.9f}".format(c), "acc =","{:.9f}".format(accu))
        loss_batch += c / total_batch
        accuracy_batch += accu / total_batch

    learningRate = sess.run(tf.compat.v1.assign(l_r, 0))

    test_total_batch = X_test.shape[0] // batchSize
    x_test = X_test.reshape(-1, batchSize, 32, 32, 3)
    y_test = y_test.reshape(-1, batchSize, 10)

    for i in range(test_total_batch):
        t_c, t_accu = sess.run([cost, accuracy], feed_dict={x: x_test[i], y:y_test[i]})
        test_loss_batch += t_c / test_total_batch
        test_accuracy_batch += t_accu / test_total_batch
        
    learningRate = sess.run(tf.compat.v1.assign(l_r, thislearningRate))

    print("Test:","%04d"%(epoch), "cost =","{:.9f}".format(test_loss_batch), "acc =","{:.9f}".format(test_accuracy_batch))
    print("Epoch:","%04d"%(epoch), "cost =","{:.9f}".format(loss_batch), "acc =","{:.9f}".format(accuracy_batch))

#    if epoch % 40 == 0: saver.save(sess, 'mymodel-epoch-{}'.format(epoch))
    if max_acc < test_accuracy_batch:
        saver.save(sess, 'max_acc-epoch-{}-{}'.format(epoch, test_accuracy_batch))
        max_acc = test_accuracy_batch

    loss_list.append(loss_batch)
    accuracy_list.append(accuracy_batch)
    test_loss_list.append(test_loss_batch)
    test_accuracy_list.append(test_accuracy_batch)

#End model
saver.save(sess, 'finish')

# --Test--

print('Learning Finished!')
plt.title("loss")
plt.plot(loss_list, 'r')
plt.plot(test_loss_list, 'b')
plt.savefig('./loss_epoch{}_batch{}.png'.format(num_epochs, batchSize))
plt.clf()

plt.title("acc")
plt.plot(accuracy_list, 'r')
plt.plot(test_accuracy_list, 'b')
plt.savefig('./acc_epoch{}_batch{}.png'.format(num_epochs, batchSize))
plt.clf()
