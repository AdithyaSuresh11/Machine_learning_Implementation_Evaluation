# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:52:31 2021

@author: adithya
"""

import numpy as np

def create_fold_files():
    filename = "W100MTimes.txt";
    data = np.loadtxt(filename);

    fold1 = data[1:5,:];
    fold2 = data[5:9,:];
    fold3 = data[9:13,:];
    fold4 = data[13:17,:];
    fold5 = data[17:21,:];
    
    np.savetxt('Fold1.txt', fold1, fmt="%0.2f",delimiter='\t',comments = "");
    np.savetxt('Fold2.txt', fold2, fmt="%0.2f",delimiter='\t',comments = "");
    np.savetxt('Fold3.txt', fold3, fmt="%0.2f",delimiter='\t',comments = "");
    np.savetxt('Fold4.txt', fold4, fmt="%0.2f",delimiter='\t',comments = "");
    np.savetxt('Fold5.txt', fold5, fmt="%0.2f",delimiter='\t',comments = "");
    
    with open('Fold1.txt') as fp:
        fold1_data = fp.read();
    with open ('Fold2.txt') as fp:
        fold2_data = fp.read();
    with open ('Fold3.txt') as fp:
        fold3_data = fp.read();
    with open ('Fold4.txt') as fp:
        fold4_data = fp.read();
    with open ('Fold5.txt') as fp:
        fold5_data = fp.read();
        
    data1234 = fold1_data + fold2_data + fold3_data + fold4_data;
    data1235 = fold1_data + fold2_data + fold3_data + fold5_data;
    data1245 = fold1_data + fold2_data + fold4_data + fold5_data;
    data1345 = fold1_data + fold3_data + fold4_data + fold5_data;
    data2345 = fold2_data + fold3_data + fold4_data + fold5_data;
    
    fold_headers_without5 = '16\t1';
    fold_headers_with5 = '15\t1';
    
    fold_headers_test_without5 = '4\t1';
    fold_headers_test_with5 = '3\t1';
    
    with open ('Fold1234.txt','w') as fp:
        fp.write(fold_headers_without5 + "\n" + data1234);
    with open ('Fold1235.txt','w') as fp:
        fp.write(fold_headers_with5 + "\n" + data1235);
    with open ('Fold1245.txt','w') as fp:
        fp.write(fold_headers_with5 + "\n" + data1245);
    with open ('Fold1345.txt','w') as fp:
        fp.write(fold_headers_with5 + "\n" + data1345);
    with open ('Fold2345.txt','w') as fp:
        fp.write(fold_headers_with5 + "\n" + data2345);
        
    with open ('Fold5_test.txt','w') as fp:
        fp.write(fold_headers_test_with5 + "\n" + fold5_data);
    with open ('Fold4_test.txt','w') as fp:
        fp.write(fold_headers_test_without5 + "\n" + fold4_data);
    with open ('Fold3_test.txt','w') as fp:
        fp.write(fold_headers_test_without5 + "\n" + fold3_data);
    with open ('Fold2_test.txt','w') as fp:
        fp.write(fold_headers_test_without5 + "\n" + fold2_data);
    with open ('Fold1_test.txt','w') as fp:
        fp.write(fold_headers_test_without5 + "\n" + fold1_data);

def train_model():
    global w_train_linear;
    global w_train_quadratic;
    global w_train_cubic;
    global input_fold;
    
    input_fold = input('Enter the fold order for training: ');
    
    if input_fold == "1234":
        training_file = 'Fold1234.txt';
    
    elif input_fold == "1235":
        training_file = 'Fold1235.txt';
    
    elif input_fold == "1245":
        training_file = 'Fold1245.txt';
        
    elif input_fold == "1345":
        training_file = 'Fold1345.txt';
    
    elif input_fold == "2345":
        training_file = 'Fold2345.txt';
    
    file = open(training_file, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t");
    train_row = int(float(splitlines[0])); # total number of data rows
    train_column = int(float(splitlines[1])); # total number of data columns except features

    training_data_linear = np.zeros([train_row,train_column+1]); # array of ones to store the rows and columns from .txt file

    # TO FIND Quadratic and Cubic solutions add column of x^2 and x^3
    for k in range(train_row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
    
        for j in range(train_column+1): # iterating through columns including features
            training_data_linear[k,j] = float(t[j]); # now data contains the plot data

    X_train_lin = np.ones([train_row,train_column+1]);
    
    for i in range(train_column):
        X_train_lin[:,i+1] = training_data_linear[:,i];

    y_train_lin = np.zeros([train_row,1]);
    y_train_lin[:,0] = training_data_linear[:,train_column];

    A_train = np.linalg.pinv(np.dot(X_train_lin.T, X_train_lin));
    B_train = np.dot(X_train_lin.T, y_train_lin);

    w_train_linear = np.dot(A_train, B_train);
    m_train_lin = X_train_lin.shape[0]

    A_train_lin = np.dot(X_train_lin, w_train_linear) - y_train_lin
    J_train_lin = (1/m_train_lin)*np.dot(A_train_lin.T, A_train_lin)
    print("J value of linear the training dataset is ",J_train_lin.item(0))
    # prediction_linear = w_train_linear[0] + np.multiply(w_train_linear[1], X_train_lin[:,1] );
    # print("Heuristic Prediction during linear training are: ", prediction_linear)

    # print("Computed weights during linear training are: ", w_train_linear);
    
##########################################################
    
    X_train_square = np.square(X_train_lin[:,1]);
    X_train_quadratic = np.insert(X_train_lin,2,X_train_square, axis = 1);
    
    A_train_quadratic = np.linalg.pinv(np.dot(X_train_quadratic.T, X_train_quadratic));
    B_train_quadratic = np.dot(X_train_quadratic.T, y_train_lin);

    w_train_quadratic = np.dot(A_train_quadratic, B_train_quadratic);
    # print("Computed weights during quadratic training are: ", w_train_quadratic);
    m_train_quad = X_train_quadratic.shape[0]

    A_train_quad = np.dot(X_train_quadratic, w_train_quadratic) - y_train_lin
    J_train_quad = (1/m_train_quad)*np.dot(A_train_quad.T, A_train_quad)
    print("J value of the quadratic training dataset is ",J_train_quad.item(0))
    
##########################################################
    
    X_train_pow3 = np.power(X_train_lin[:,1],3);
    X_train_cubic = np.insert(X_train_quadratic,3,X_train_pow3, axis = 1);
    
    A_train_cubic = np.linalg.pinv(np.dot(X_train_cubic.T, X_train_cubic));
    B_train_cubic = np.dot(X_train_cubic.T, y_train_lin);

    w_train_cubic = np.dot(A_train_cubic, B_train_cubic);
    # print("Computed weights during cubic training are: ", w_train_cubic);
    m_train_cube = X_train_cubic.shape[0]

    A_train_cube = np.dot(X_train_cubic, w_train_cubic) - y_train_lin
    J_train_cube = (1/m_train_cube)*np.dot(A_train_cube.T, A_train_cube)
    print("J value of the cubic training dataset is ",J_train_cube.item(0))
    return input_fold
    

def test_model():
    if input_fold == "1234": 
        testing_file = "Fold5_test.txt";
    
    elif input_fold == "1235": 
        testing_file = "Fold4_test.txt";
        
    elif input_fold == "1245": 
        testing_file = "Fold3_test.txt";
        
    elif input_fold == "1345": 
        testing_file = "Fold2_test.txt";
        
    elif input_fold == "2345": 
        testing_file = "Fold1_test.txt";
    
    file = open(testing_file, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t"); 
    test_row = int(float(splitlines[0])); # total number of data rows
    test_column = int(float(splitlines[1])); # total number of data columns except features
    
    testing_data = np.zeros([test_row,test_column+1]); # array of ones to store the rows and columns from .txt file

    for k in range(test_row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
        for j in range(test_column+1): # iterating through columns including features
            testing_data[k,j] = float(t[j]); # now data contains the plot data
    
    X_test = np.ones([test_row,test_column+1])
    for i in range(test_column):
        X_test[:,i+1] = testing_data[:,i]
    y_test = np.zeros([test_row,1])
    y_test[:,0] = testing_data[:,test_column]
        
    A_test = np.linalg.pinv(np.dot(X_test.T, X_test))
    B_test = np.dot(X_test.T, y_test)
    m_test = X_test.shape[0]

    A_test = np.dot(X_test, w_train_linear) - y_test
    J_test = (1/m_test)*np.dot(A_test.T, A_test)
    print("J value of the linear testing dataset",J_test.item(0))
    
########################################################
    
    X_test_square = np.square(X_test[:,1]);
    X_test_quadratic = np.insert(X_test,2,X_test_square, axis = 1);
    
    A_test_quad = np.linalg.pinv(np.dot(X_test_quadratic.T, X_test_quadratic))
    B_test_quad = np.dot(X_test_quadratic.T, y_test)
    m_test_quad = X_test_quadratic.shape[0]

    A_test_quad = np.dot(X_test_quadratic, w_train_quadratic) - y_test
    J_test_quad = (1/m_test_quad)*np.dot(A_test_quad.T, A_test_quad)
    print("J value of the quadratic testing dataset",J_test_quad.item(0))

########################################################
    
    X_test_pow3 = np.power(X_test[:,1],3);
    X_test_cubic = np.insert(X_test_quadratic,3,X_test_pow3, axis = 1);
    
    A_test_cube = np.linalg.pinv(np.dot(X_test_cubic.T, X_test_cubic))
    B_test_cube = np.dot(X_test_cubic.T, y_test)
    m_test_cube = X_test_cubic.shape[0]

    A_test_cube = np.dot(X_test_cubic, w_train_cubic) - y_test
    J_test_cube = (1/m_test_cube)*np.dot(A_test_cube.T, A_test_cube)
    print("J value of the cubic testing dataset",J_test_cube.item(0))
    
    
if __name__ == "__main__":
    create_fold_files();
    train_model();
    test_model();
    # create_fold_files();
    # train_model(); # the train_model function is called in main function