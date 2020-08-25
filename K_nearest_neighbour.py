# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:59:55 2020

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
x1=np.random.rand(num_observation)
x2=np.random.rand(num_observation)


y=np.zeros((num_observation))
y[x1>0.5]=1

#visualise data
plt.figure()
plt.title("visualise data")
plt.scatter(x1[y==0],x2[y==0],color="blue",label="0")
plt.scatter(x1[y==1],x2[y==1],color="red",label="1")
plt.show()

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
y_train=y[index_list][:num_train]
y_test=y[index_list][num_train:]

########################################################
########################################################
########################################################
#KNN
K=5

def KNN_classifier(new_x1,new_x2):
    #find euclidean distance
    new_point=np.array([new_x1,new_x2])[:,np.newaxis]
    old_point=np.array([x1_train,x2_train])
    distance_list=np.linalg.norm(new_point-old_point,axis=0)
    
    #select only K closest point
    index_list=[]
    for i in range(0,K):
        index=np.argmin(distance_list)
        index_list.append(index)
        distance_list[index]=999999
    
    #classification
    category_list=y_train[index_list]
    cat1_score=0
    cat2_score=0
    
    for cat in category_list:
        if cat==0:
            cat1_score+=1
        else:
            cat2_score+=1
    if cat1_score>cat2_score:
        pred_y=0
    elif cat2_score>cat1_score:
        pred_y=1
    else:
        pred_y=np.random.randint(0,2)
    
    return pred_y
    
    
#see training set accuracy
pred_y_train=[]

for new_x1,new_x2 in zip(x1_train,x2_train):
    
    pred_y_train.append(KNN_classifier(new_x1,new_x2))

pred_y_train=np.array(pred_y_train)
accuracy_train=np.zeros((num_train))
accuracy_train[pred_y_train==y_train]=1
accuracy_train=accuracy_train.sum()/num_train

print("accuracy training set : ",accuracy_train)

#see test set accuracy
pred_y_test=[]

for new_x1,new_x2 in zip(x1_test,x2_test):
    
    pred_y_test.append(KNN_classifier(new_x1,new_x2))

pred_y_test=np.array(pred_y_test)
accuracy_test=np.zeros((num_test))
accuracy_test[pred_y_test==y_test]=1
accuracy_test=accuracy_test.sum()/num_test

print("accuracy test set : ",accuracy_test)
