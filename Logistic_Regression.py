# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:47:30 2020

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
category=2
num_observation=500
x=np.linspace(0,100,num_observation)
y=np.zeros((num_observation))
y[x>50]=1

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
#Logistic Regression

sess=tf.Session()
pred_coeff=tf.Variable(0,dtype=tf.float64)
pred_intercept=tf.Variable(0,dtype=tf.float64)

linear_y=(pred_coeff*x_train)+pred_intercept
logistic_line=1/(1+tf.exp(-linear_y))
pred_y_train=tf.math.round(logistic_line)
accuracy=tf.constant(y_train)==pred_y_train

#gradient descent
init=tf.initialize_all_variables()
sess.run(init)
pred_y_train=sess.run(pred_y_train)

