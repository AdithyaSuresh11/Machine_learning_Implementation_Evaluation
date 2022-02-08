# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:31:30 2021

@author: adithya
"""

import numpy as np
from matplotlib import pyplot as plt

### For dataset name - use - P4train.txt and P4test.txt

def plot():
    file_train = "P4train.txt" 
    file_train = open(file_train,'r');
    stringtoint = file_train.readline(); 
    splitlines = stringtoint.split("\t"); 
    row = int(splitlines[0]); # total number of data rows
    column = int(splitlines[1]); # total number of data columns except features
    data_train = np.zeros([row,column+1]); # array of ones to store the rows and columns from .txt file
    for k in range(row): # iterating through the rows
        aString = file_train.readline();
        t = aString.split("\t");
        for j in range(column+1): # iterating through columns including features
            data_train[k,j] = float(t[j]); # now data contains the plot data
    
    file_test = "P4test.txt"
    file_test = open(file_test,'r');
    stringtoint_test = file_test.readline(); 
    splitlines_test = stringtoint_test.split("\t"); 
    row_test = int(splitlines_test[0]); # total number of data rows
    column_test = int(splitlines_test[1]); # total number of data columns except features
    data_test = np.zeros([row_test,column_test+1]); # array of ones to store the rows and columns from .txt file
    for k in range(row_test): # iterating through the rows
        aString_test = file_test.readline();
        t_test = aString_test.split("\t");
        for j in range(column_test+1): # iterating through columns including features
            data_test[k,j] = float(t_test[j]); # now data contains the plot data
    total_plot = np.concatenate((data_train, data_test));
    total_plot_row = len(total_plot);
    total_plot_column = len(total_plot[0]);
    for line in range(total_plot_row):
        coloring = total_plot[line,2];
        colors = ['r','g'];
        if coloring == 0:
            failed = plt.scatter(total_plot[line,0],total_plot[line,1],color = colors[0],marker = 'x');
        if coloring == 1:
            passed = plt.scatter(total_plot[line,0],total_plot[line,1],color = colors[1],marker = 'o');
    plt.xlabel('Test 1 results');
    plt.ylabel('Test 2 results');
    plt.legend((failed,passed),('Failed','Passed'),loc = "upper right");
    plt.title("Test result comparison plot");
    plt.savefig("result.png",bbox_inches="tight");
    plt.show();

def logistic_train():
    global w_train
    # file_train = "P4train.txt" 
    file_train = input("Enter the name of your training data file: ");
    file_train = open(file_train,'r');
    stringtoint = file_train.readline();
    splitlines = stringtoint.split("\t");
    row = int(splitlines[0]); # total number of data rows
    column = int(splitlines[1]); # total number of data columns except features
    data_train = np.zeros([row,column+1]); # array of ones to store the rows and columns from .txt file
    
    for k in range(row): # iterating through the rows
        aString = file_train.readline();
        t = aString.split("\t");
        for j in range(column+1): # iterating through columns including features
            data_train[k,j] = float(t[j]); # now data contains the plot data
    X_train = np.ones([row,column+1]);
    
    for i in range(column):
        X_train[:,i+1] = data_train[:,i] # Taking the x1 (second column) values
    
    y_train = np.zeros([row,1])
    y_train[:,0] = data_train[:,column] # Taking the y 
    
    with open("P4train.txt") as fp:
        data_write = fp.read();
    with open('Suresh_Adithya_P4train.txt','w') as fp:
        xpower = 4;
        thePower = 3;
        counter = 0;
        header = '85\t20';
        fp.write(header+'\n');
        for k in range(row):
            for j in range(xpower+1):
                for i in range(thePower+1):
                    temp = (X_train[:,1]**i)*(X_train[:,2]**j)
                    if (temp[i] != 1):
                        fp.write(str(temp[k])+"\t")
            fp.write(str(y_train[:,0][k])+"\n")
        fp.close()
            
    file_t = "Suresh_Adithya_P4train.txt"
    file_t = open(file_t,'r');
    stringto = file_t.readline(); 
    splitline = stringto.split("\t"); 
    row_t = int(splitline[0]); # total number of data rows
    column_t = int(splitline[1]); # total number of data columns except features
    data_tr= np.zeros([row_t,column_t]); # array of ones to store the rows and columns from .txt file

    for k in range(row_t): # iterating through the rows
        aSt = file_t.readline();
        tra = aSt.split("\t");
        # print(tra)
        for j in range(column_t): # iterating through columns including features
            data_tr[k,j] = float(tra[j]); # now data contains the plot data
    
    X_train = np.ones([row_t,column_t+1]);
    
    for i in range(column_t):
        X_train[:,i+1] = data_tr[:,i] # Taking the x1 (second column) values
    
    X_train = np.delete(X_train,column_t,axis = 1)
    y_train = np.zeros([row_t,1])
    y_train[:,0] = data_tr[:,column_t-1] # Taking the y 
    
    w_train = np.zeros([column_t,1]);
    ones_vec = np.ones([1,row_t]);
    m = X_train[:,0];
    iterations = 100000;
    alpha = 0.5;

    empty = [];
    for i in range(iterations):
        hypothesis = 1/(1+np.exp(-np.dot(X_train,w_train)));
        cost = -y_train * np.log(hypothesis) - (1-y_train) * np.log(1-hypothesis);
        j = (1/len(m)) * np.dot(ones_vec,cost)
        empty.append(j)
        temp1 = (alpha/len(m)) * np.dot(np.subtract(hypothesis,y_train).T,X_train);
        w_train = w_train - (temp1.T);
        
    print('Final Weights computed after training: ',w_train)
    print('Total number of iterations involved: ',iterations)
    print("Final J after training is: ",empty[iterations-1][0][0])
    J = np.reshape(empty,(iterations,1));
    iterations = list(range(1,len(empty)+1));
    plt.title("Iterations vs J_w for training")
    plt.xlabel("Number of Iterations of Gradient Descent")
    plt.ylabel("J")
    plt.plot(iterations,J)
    return w_train    

def logistic_test():
    # file_test = "P4test.txt"
    file_test = input("Enter the name of your testing data file: ");
    file_test = open(file_test,'r');
    stringtoint = file_test.readline(); 
    splitlines = stringtoint.split("\t"); 
    row = int(splitlines[0]); # total number of data rows
    column = int(splitlines[1]); # total number of data columns except features
    data_test = np.zeros([row,column+1]); # array of ones to store the rows and columns from .txt file
    for k in range(row): # iterating through the rows
        aString = file_test.readline();
        t = aString.split("\t");
        for j in range(column+1): # iterating through columns including features
            data_test[k,j] = float(t[j]); # now data contains the plot data 
    
    X_test = np.ones([row,column+1]);
    
    for i in range(column):
        X_test[:,i+1] = data_test[:,i] # Taking the x1 (second column) values
    
    y_test = np.zeros([row,1])
    y_test[:,0] = data_test[:,column] # Taking the y 

    
    with open("P4test.txt") as fp:
        data_write = fp.read();
    with open('Suresh_Adithya_P4test.txt','w') as fp:
        xpower = 4;
        thePower = 3
        counter = 0;
        header = '33\t20';
        fp.write(header+'\n');
        for k in range(row):
            for j in range(xpower+1):
                for i in range(thePower+1):
                    temp = (X_test[:,1]**i)*(X_test[:,2]**j)
                    if (temp[i] != 1):
                        fp.write(str(temp[k])+"\t")
            fp.write(str(y_test[:,0][k])+"\n")
        fp.close()
            
    file_te = "Suresh_Adithya_P4test.txt"
    file_te = open(file_te,'r');
    stringto_te = file_te.readline(); 
    splitline_te = stringto_te.split("\t"); 
    row_te = int(splitline_te[0]); # total number of data rows
    column_te = int(splitline_te[1]); # total number of data columns except features
    data_te= np.zeros([row_te,column_te]); # array of ones to store the rows and columns from .txt file

    for k in range(row_te): # iterating through the rows
        aSte = file_te.readline();
        trae = aSte.split("\t");
        for j in range(column_te): # iterating through columns including features
            data_te[k,j] = float(trae[j]); # now data contains the plot data
    X_test = np.ones([row_te,column_te+1]);
    
    for i in range(column_te):
        X_test[:,i+1] = data_te[:,i] # Taking the x1 (second column) values
    
    X_test = np.delete(X_test,column_te,axis = 1)
    y_test = np.zeros([row_te,1])
    y_test[:,0] = data_te[:,column_te-1] # Taking the y
    

    ones_vec_test = np.ones([1,row_te]);
    m_test = X_test[:,0];
    
    hypothesis_test = 1/(1+np.exp(-np.dot(X_test,w_train)));
    cost_test = -y_test * np.log(hypothesis_test) - (1-y_test) * np.log(1-hypothesis_test)
    j_test = (1/len(m_test)) * np.dot(ones_vec_test,cost_test)
    y_pred = np.dot(X_test, w_train);
    for i in range(len(y_pred)):
        if (y_pred[i] < 0.5): y_pred[i]= 0;
        elif (y_pred[i] > 0.5): y_pred[i]= 1;
    print("Final J after testing is",j_test[0][0])
    
    true_neg = 0;
    true_pos = 0;
    false_pos = 0;
    false_neg = 0;
    
    for i in range(len(y_test)):
        if (y_test[i]==1 and y_pred[i]==1):
            true_pos += 1;
        if (y_test[i]==1 and y_pred[i]==0):
            false_neg += 1;
        if (y_test[i]==0 and y_pred[i]==1):
            false_pos += 1;
        if (y_test[i]==0 and y_pred[i]==0):
            true_neg += 1;
    print("True negative is: ", true_neg)
    print("True positive is: ", true_pos)
    print("False negative is: ", false_neg)
    print("False positive is: ", false_pos)
    confusion_matrix = np.array([[true_neg, false_pos],[false_neg, true_pos]]);
    print("The Confusion Matrix is: ",confusion_matrix)
    accuracy = ((true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg))*100 ;
    precision = (true_pos/(true_pos + false_pos))*100;
    recall = (true_pos/(true_pos + false_neg))*100;
    f1 = (2 * precision * recall)/(precision + recall)
    print("The accuracy in % is: ",accuracy)
    print("The precision in % is: ",precision)
    print("The recall in % is: ",recall)
    print("The F1 score in % is: ",f1)

if __name__ == "__main__":
    plot();
    logistic_train();
    logistic_test();