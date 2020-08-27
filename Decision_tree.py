# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:24:02 2020

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
#entropy calculation
def entropy_calculator(any_list):

    entropy=0
    for i in range(0,category):
        p=np.zeros(any_list.shape)
        p[any_list==i]=1
        if p.sum()!=0:
            p=p.sum()/any_list.shape[0]
            entropy+=-p*np.log(p)
    return entropy


########################################################
########################################################
########################################################
#consider good split | got pure one and zero after split
#before split
b=np.copy(y)
num_b=num_observation

#after split
a1=np.ones((int(num_observation/2)))
a2=np.zeros(int((num_observation/2)))
num_a1=int(num_observation/2)
num_a2=int(num_observation/2)

#calculate entropy
entropy_before=entropy_calculator(b)
entropy_after=(entropy_calculator(a1)*num_a1/num_b)+(entropy_calculator(a2)*num_a2/num_b)

#calculare information gain
information_gain=entropy_before-entropy_after
print("good split information gain: ",information_gain)
    
########################################################
########################################################
########################################################
#consider bad split | still got one and zero in the same set
#before split
b=np.copy(y)
num_b=num_observation

#after split
index_list=np.arange(0,num_b)
np.random.shuffle(index_list)
a1=b[index_list[:int(num_b/2)]]
a1=b[index_list[int(num_b/2):]]


#calculate entropy
entropy_before=entropy_calculator(b)
entropy_after=(entropy_calculator(a1)*num_a1/num_b)+(entropy_calculator(a2)*num_a2/num_b)

#calculare information gain
information_gain=entropy_before-entropy_after
print("bad split information gain: ",information_gain)

########################################################
########################################################
########################################################
#real data set


