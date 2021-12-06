import numpy as np
from numpy.core.function_base import _logspace_dispatcher
from numpy.core.numeric import outer
import pandas as pd
import tensorflow as tf
import logging 
logging.basicConfig(level=logging.DEBUG) 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def multilayer_perceptron(input_d): 
    #input_d : input feature 
    #sigmoid : f(x) = 1 / (1 + e^{-x})。
    #Task of neurons of first hidden layer  (?,6) * (6,10) + (10,)
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))  #input_d * w1 + b1
    #Task of neurons of second hidden layer  () * (10,10) + (10,)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2)) 
    #Task of neurons of output layer () * (10,1) + (1,)
    out_layer = tf.add(tf.matmul(layer_2, w3),b3) 

    return out_layer

#Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 6
n_output = 2
#Learning parameters
learning_constant = 0.1
number_epochs = 1000
#batch_size = 4000

column_names=['IOP', 'PI', 'LLA', 'SS','PR','DOLS','Category']
data =pd.read_csv("/Users/mengjiao/Documents/MachineLearning/MLGroup24/Lab3/Classification/column_2C.csv",names=column_names)
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)
train_labels_tmp = train_data.pop('Category')
test_labels_tmp = test_data.pop('Category')
train_labels = pd.get_dummies(train_labels_tmp)
test_labels = pd.get_dummies(test_labels_tmp)
trainStats = train_data.describe()
trainStats = trainStats.transpose() 
TrainData = (train_data-trainStats["mean"])/trainStats["std"]
TestData = (test_data-trainStats["mean"])/trainStats["std"]

train_labels=pd.DataFrame(train_labels)
test_labels =pd.DataFrame(test_labels)

#Defining the input and the output
X = tf.placeholder('float', [None, n_input])
Y = tf.placeholder('float', [None, n_output])
#DEFINING WEIGHTS AND BIASES
#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))  #random_normal()服从指定正态分布的序列”中随机取出指定个数的值
#Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2])) 
#Biases output layer
b3 = tf.Variable(tf.random_normal([n_output]))
#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1])) 
#Weights connecting first hidden layer with second hidden layer 
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2])) 
#Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
#loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)


#Initializing the variables
init = tf.global_variables_initializer()

plt.ion()
fig, (ax , ax1)= plt.subplots(1,2, figsize=(8, 4))
accuracies, steps, losstest, losstrain = [], [], [], []
with tf.Session() as sess:
    sess.run(init)
    #Training epoch
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: TrainData, Y: train_labels})
        #Display the epoch
        if epoch % 100 == 0:
            y_pred = sess.run(tf.argmax(neural_network, 1), feed_dict={X: TestData})
            y_orig = sess.run(tf.argmax(Y, 1), feed_dict={Y: test_labels})

            preds_check = tf.equal(y_pred, y_orig)
            accuracy_op = tf.reduce_mean(tf.cast(preds_check, tf.float32))
            accuracy_score = sess.run(accuracy_op)
            loss_test = sess.run(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=neural_network), feed_dict={X: TestData,Y: test_labels})
            loss_train = sess.run(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=neural_network), feed_dict={X: TrainData,Y: train_labels})
            steps.append(epoch)
            accuracies.append(accuracy_score)
            losstest.append(loss_test)
            losstrain.append(loss_train)
            print(f"epoch {epoch} | accuracy={accuracy_score} | loss={loss_test}")
            ax.cla()
            ax.plot(steps, accuracies, label="accuracy")
            ax.set_ylim(ymax=1)
            ax.set_ylabel("accuracy")

            ax1.cla()
            ax1.plot(steps, losstest, label="Valid Error")
            ax1.plot(steps, losstrain, label="Train Error")
            ax1.set_ylim(ymax=1)
            ax1.set_ylabel("LOSS")
            ax1.set_xlabel("Epoch")
            plt.pause(0.01)
    plt.ioff()
    plt.show()


'''
    print("===============10-fold cross validation")
    k=10
    datax=train_data
    datay=train_labels
    acc = 0
    for train_index,test_index in KFold(k,shuffle=True).split(datax):
        
        x_train, x_test=datax.iloc[train_index],datax.iloc[test_index]
        y_train, y_test=datay.iloc[train_index],datay.iloc[test_index]

        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})

        y_pred = sess.run(tf.argmax(neural_network, 1), feed_dict={X: x_test})
        y_orig = sess.run(tf.argmax(Y, 1), feed_dict={Y: y_test})

        preds_check = tf.equal(y_pred, y_orig)
        accuracy_op = tf.reduce_mean(tf.cast(preds_check, tf.float32))
        accuracy_score = sess.run(accuracy_op)
        acc += accuracy_score
        print(" Accurate: %.2f" % accuracy_score)
    average_acc = acc/10
    print(" Average Accurate: %.2f" % average_acc)

'''