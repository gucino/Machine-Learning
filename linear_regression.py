# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:10:04 2020

@author: Tisana
"""

########################################################
########################################################
########################################################
#import library
import numpy as np
import matplotlib.pyplot as plt

#generate data
num_observation=500
x=np.linspace(1,100,num_observation)
noise_list=np.random.rand(num_observation)*100
true_coeff=4.85
y=(true_coeff*x)+noise_list

#visualise data
plt.figure()
plt.title("visualise data")
plt.scatter(x,y,s=1)
plt.show()

#split train and test
percent_train=0.8
num_train=int(num_observation*percent_train)
num_test=num_observation-num_train

index_list=np.arange(0,num_observation)
np.random.shuffle(index_list)

x_train=x[index_list][:num_train]
x_test=x[index_list][num_train:]
y_train=y[index_list][:num_train]
y_test=y[index_list][num_train:]

########################################################
########################################################
########################################################
#linear regression

import tensorflow as tf
sess=tf.Session()



pred_coeff=tf.Variable(1,dtype=tf.float64)
pred_y_train=pred_coeff*x_train
loss=tf.reduce_mean((pred_y_train-y_train)**2)
learning_rate=0.0001
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#gradient descent
init=tf.initialize_all_variables()
sess.run(init)
sess.run(pred_coeff)
print("before gradient descent : ",sess.run(loss))
loss_list=[]
for i in range(0,10000):
    l=sess.run(loss)
    loss_list.append(l)
    sess.run(train)
    
    if i%100==0:
        pass
        #print(l)
   
print("after gradient descent : ",sess.run(loss))
plt.figure()
plt.title("error during gradient descent")
plt.plot(np.arange(0,len(loss_list)),loss_list)

best_coeff=sess.run(pred_coeff)
print("best coeff : ",best_coeff)

#prediction
pred_y_train=best_coeff*x_train
pred_y_test=best_coeff*x_test

#visualise training set
plt.figure()
plt.title("visualise training set")
plt.scatter(x_train,y_train,color="blue",s=1)
plt.plot(x_train,pred_y_train,color="red")
plt.show()

#visulise test set
plt.figure()
plt.title("visualise test set")
plt.scatter(x_test,y_test,color="blue",s=1)
plt.plot(x_test,pred_y_test,color="red")
plt.show()


