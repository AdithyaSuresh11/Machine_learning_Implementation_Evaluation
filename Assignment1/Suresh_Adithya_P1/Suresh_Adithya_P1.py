# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 07:28:43 2021

@author: adithya
"""

import numpy as np

####### COURSE: MACHINE LEARNING: IMPLEMENTATION AND EVALUATION ########
####### AUTHOR: ADITHYA SURESH, C18590622 ##########
####### PROJECT 1 ########

### For typing the dataset name, please use - TrainCubed.txt and TestCubed.txt ###

### Numpy library is imported for use ###

def train_model():
    global w_train;
    training_file = input("Enter the name of your training data file: ");
    file = open(training_file, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t"); 
    train_row = int(splitlines[0]); # total number of data rows
    train_column = int(splitlines[1]); # total number of data columns except features
    training_data = np.zeros([train_row,train_column+1]); # array of ones to store the rows and columns from .txt file

    for k in range(train_row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
        for j in range(train_column+1): # iterating through columns including features
            training_data[k,j] = float(t[j]); # now data contains the plot data

    X_train = np.ones([train_row,train_column+1]);
    
    for i in range(train_column):
        X_train[:,i+1] = training_data[:,i] # Taking the x1 (second column) values

    y_train = np.zeros([train_row,1])
    y_train[:,0] = training_data[:,train_column] # Taking the y values
    
    A_train = np.linalg.pinv(np.dot(X_train.T, X_train))
    B_train = np.dot(X_train.T, y_train)

    w_train = np.dot(A_train, B_train) # Weights calculation
    m_train = X_train.shape[0]

    A_train = np.dot(X_train, w_train) - y_train
    J_train = (1/m_train)*np.dot(A_train.T, A_train) # Jw(x) calculation for training
    print("Computed weights during training are: ", w_train)
    print("J value of the training dataset is ",J_train.item(0))
    return w_train

def test_model():
    testing_file = input("Enter the name of your testing data file: ");
    file = open(testing_file, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t"); 
    test_row = int(splitlines[0]); # total number of data rows
    test_column = int(splitlines[1]); # total number of data columns except features
    
    testing_data = np.zeros([test_row,test_column+1]); # array of ones to store the rows and columns from .txt file

    for k in range(test_row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
        for j in range(test_column+1): # iterating through columns including features
            testing_data[k,j] = float(t[j]); # now data contains the plot data
    
    X_test = np.ones([test_row,test_column+1])
    for i in range(test_column):
        X_test[:,i+1] = testing_data[:,i]  # Taking the x1 (second column) values
    y_test = np.zeros([test_row,1])
    y_test[:,0] = testing_data[:,test_column] # Taking the y values
    
    A_test = np.linalg.pinv(np.dot(X_test.T, X_test))
    B_test = np.dot(X_test.T, y_test)
    m_test = X_test.shape[0]

    A_test = np.dot(X_test, w_train) - y_test
    J_test = (1/m_test)*np.dot(A_test.T, A_test) # Jw(x) calculation for testing
    print("J value of the testing dataset is ",J_test.item(0))
    
             
if __name__ == "__main__":
    train_model(); # the train_model function is called in main function
    test_model(); # the test_model function is called in main function