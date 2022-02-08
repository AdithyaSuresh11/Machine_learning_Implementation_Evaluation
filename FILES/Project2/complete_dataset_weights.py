# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:35:58 2021

@author: adithya
"""

import numpy as np

### Numpy library is imported for use ###

def train_model():
    # training_file = input("Enter the name of your training data file: ");
    training_file = "W100MTimes.txt"
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

    X_train_square = np.square(X_train[:,1]);
    X_train = np.insert(X_train,2,X_train_square, axis = 1);
    
    y_train = np.zeros([train_row,1])
    y_train[:,0] = training_data[:,train_column] # Taking the y values
    
    A_train = np.linalg.pinv(np.dot(X_train.T, X_train))
    B_train = np.dot(X_train.T, y_train)

    w_train = np.dot(A_train, B_train) # Weights calculation
    
    print('Computed Weights for the complete W100MTimes.txt dataset are: ', w_train)
    return w_train
    

if __name__ == "__main__":
    train_model(); # the train_model function is called in main function