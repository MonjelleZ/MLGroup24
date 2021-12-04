import numpy as np
import pandas as pd
import math
import random
import tensorflow as tf
import logging 
logging.basicConfig(level=logging.DEBUG) 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
learning_constant = 0.2
number_epochs = 4000
#batch_size = 4000

column_names=['IOP', 'PI', 'LLA', 'SS','PR','DOLS','Category']
data =pd.read_csv("/Users/mengjiao/Documents/MachineLearning/MLGroup24/Lab3/column_2C.csv",names=column_names)
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)
train_labels_tmp = train_data.pop('Category')
test_labels_tmp = test_data.pop('Category')
train_labels = pd.get_dummies(train_labels_tmp)
test_labels = pd.get_dummies(test_labels_tmp)
trainStats = train_data.describe()
trainStats = trainStats.transpose()
def Norm(x):
    return (x-trainStats["mean"])/trainStats["std"]
 
TrainData = Norm(train_data)
TestData = Norm(test_data)

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
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)


#Initializing the variables
init = tf.global_variables_initializer()

accuracies, steps = [], []
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
            print("epoch {0:04d} accuracy={1:.8f}".format(epoch, accuracy_score))


    pred = (neural_network) # Apply softmax to logits 
    #accuracy=tf.keras.losses.MSE(pred,Y) 
    #print("Accuracy:", accuracy.eval({X: TrainData, Y:train_labels})) 
    #tf.keras.evaluate(pred,batch_x)
    #print("Prediction:", pred.eval({X: TrainData})) 
    #output=neural_network.eval({X: TrainData}) 
    #plt.plot(train_labels[0:10], 'ro', output[0:10], 'bo') 
    #plt.ylabel('some numbers')
    #plt.show()
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