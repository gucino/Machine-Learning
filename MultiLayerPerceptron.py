# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:19:24 2020

@author: Tisana
"""

########################################################
########################################################
########################################################
#import library
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#generate data
num_observation=500
x1=np.linspace(1,100,num_observation)
x2=np.linspace(-100,5,num_observation)
x3=np.linspace(1000,1000,num_observation)
y=(2*x1)+(4.563*x2)+(3.32*x3)

#split train and test
percent_train=0.8
num_train=int(num_observation*percent_train)
num_test=num_observation-num_train

index_list=np.arange(0,num_observation)
np.random.shuffle(index_list)

x1_train=x1[index_list][:num_train]
x1_test=x1[index_list][num_train:]
x2_train=x2[index_list][:num_train]
x2_test=x2[index_list][num_train:]
x3_train=x3[index_list][:num_train]
x3_test=x3[index_list][num_train:]
y_train=y[index_list][:num_train]
y_test=y[index_list][num_train:]
########################################################
########################################################
########################################################
#single layer pereptron

num_feature=3

#structure
input_layer=tf.placeholder(dtype=tf.float64,shape=(None,3))
weight=tf.Variable(np.array([1,1,1]),dtype=tf.float64)[:,tf.newaxis]
bias=tf.Variable(1,dtype=tf.float64)
output_layer=tf.squeeze(tf.matmul(input_layer,weight)+bias)

#train
training_data=np.einsum("ij->ji",np.array([x1_train,x2_train,x3_train]))
learning_rate=0.0000001
loss=tf.reduce_mean((output_layer-y_train)**2)
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
mse_list=[]
print("before : ",sess.run(loss,feed_dict={input_layer:training_data}))
for i in range(0,1000):
    l=sess.run(loss,feed_dict={input_layer:training_data})
    #print(l)
    mse_list.append(l)
    sess.run(train,feed_dict={input_layer:training_data})
    

print("after : ",sess.run(loss,feed_dict={input_layer:training_data}))
plt.figure()
plt.title("error per iteration")
plt.plot(np.arange(0,len(mse_list)),mse_list)
plt.show()

best_weight=sess.run(weight)
best_bias=sess.run(bias)

#training set prediction
pred_y_train=np.squeeze(np.matmul(training_data,best_weight)+best_bias)
mse_train=np.mean((pred_y_train-y_train)**2)
print("MSE train : ",mse_train)

#test set prediction
test_data=np.einsum("ij->ji",np.array([x1_test,x2_test,x3_test]))
pred_y_test=np.squeeze(np.matmul(test_data,best_weight)+best_bias)
mse_test=np.mean((pred_y_test-y_test)**2)
print("MSE test : ",mse_test)

########################################################
########################################################
########################################################
#multi layer perceptron

num_feature=3

#structure
input_layer=tf.placeholder(dtype=tf.float64,shape=(None,3))
w1=tf.Variable(np.array([[1,1,1],[1,1,1],[1,1,1]]),dtype=tf.float64)
w2=tf.Variable(np.array([1,1,1]),dtype=tf.float64)[:,tf.newaxis]
b1=tf.Variable([1,1,1],dtype=tf.float64)
b2=tf.Variable(1,dtype=tf.float64)
hidden_layer=(tf.matmul(input_layer,w1)+b1)
output_layer=tf.squeeze(tf.matmul(hidden_layer,w2)+b2)



#train
training_data=np.einsum("ij->ji",np.array([x1_train,x2_train,x3_train]))
learning_rate=0.0000001
loss=tf.reduce_mean((output_layer-y_train)**2)
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
mse_list=[]
print("before : ",sess.run(loss,feed_dict={input_layer:training_data}))
for i in range(0,1000):
    l=sess.run(loss,feed_dict={input_layer:training_data})
    #print(l)
    mse_list.append(l)
    sess.run(train,feed_dict={input_layer:training_data})
    

print("after : ",sess.run(loss,feed_dict={input_layer:training_data}))
plt.figure()
plt.title("error per iteration")
plt.plot(np.arange(0,len(mse_list)),mse_list)
plt.show()

best_w1=sess.run(w1)
best_w2=sess.run(w2)
best_b1=sess.run(b1)
best_b2=sess.run(b2)


#training set prediction
hidden_layer=(np.matmul(training_data,best_w1)+best_b1)
pred_y_train=np.squeeze(np.matmul(hidden_layer,best_w2)+best_b2)
mse_train=np.mean((pred_y_train-y_train)**2)
print("MSE train : ",mse_train)

#test set prediction
test_data=np.einsum("ij->ji",np.array([x1_test,x2_test,x3_test]))
hidden_layer=(np.matmul(test_data,best_w1)+best_b1)
pred_y_test=np.squeeze(np.matmul(hidden_layer,best_w2)+best_b2)
mse_test=np.mean((pred_y_test-y_test)**2)
print("MSE test : ",mse_test)

########################################################
########################################################
########################################################
