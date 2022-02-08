# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:31:30 2021

@author: adithya
"""

import numpy as np

### For dataset name - use - P3train.txt and P3test.txt

def load_testdata():
    global test_data, test_class
    file = input("Enter the name of your testing data file: ");
    file = open(file, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t"); 
    row = int(splitlines[0]); # total number of data rows
    column = int(splitlines[1]); # total number of data columns except features
    data = np.zeros([row,column+1]); # array of ones to store the rows and columns from .txt file
    for k in range(row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
        for j in range(column+1): # iterating through columns including features
            data[k,j] = float(t[j]); # now data contains the plot data
    
    test_data = np.zeros([row,column]);
    test_data = data[:,0:2];
    test_class = np.zeros([row,column-1])
    test_class = data[:,2:3] 
    # print(val_class)
    return test_data, test_class

def load_traindata():
    global train_data, train_class
    file = input("Enter the name of your training data file: ");
    file = open(file, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t"); 
    row = int(splitlines[0]); # total number of data rows
    column = int(splitlines[1]); # total number of data columns except features
    data = np.zeros([row,column+1]); # array of ones to store the rows and columns from .txt file
    for k in range(row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
        for j in range(column+1): # iterating through columns including features
            data[k,j] = float(t[j]); # now data contains the plot data
    
    train_data = np.zeros([row,column]);
    train_data = data[:,0:2];
    train_class = np.zeros([row,column-1])
    train_class = data[:,2:3] 
    return train_data, train_class

def distance():
    global ids, sort_distance
    a = 0;
    b = 0;
    distance = np.zeros([len(test_data),len(train_data)]);
    for i in range(len(test_data)):
        test_pointx = test_data[a,b];
        test_pointy = test_data[a,b+1];
        a += 1;
        for j in range(len(train_data)):
            distance[i,j] = ((test_pointx - train_data[j,0])**2 + (test_pointy - train_data[j,1])**2)**0.5;  
    
    ids = np.argsort(distance)
    sort_distance = np.sort(distance)
    return ids, sort_distance

def nearest_neighbor():
    global k;
    k = 5;
    # for k in range(1,25,2):
    nearest_points = np.zeros([len(sort_distance),k]);
    nearest_ids = np.zeros([len(ids),k]);
    for i in range(len(sort_distance)):
        for j in range(k):
            nearest_points[i,j] = sort_distance[i,j];
            nearest_ids[i,j] = ids[i,j];

    counter = 0;
    counter0 = 0;
    compare_class = np.zeros([len(ids),k]);
    for i in range(len(ids)):
        for j in range(k):
            adi = ids[i,j];
            compare_class[i,j] = train_class[adi,0]
    # print(compare_class)
    compare_id = list();
    compare_id0 = list();
    for i in range(len(ids)):
        counter = 0;
        counter0 = 0;
        for j in range(k):
            if (compare_class[i][j] == 1):
                counter += 1;
            if(compare_class[i,j] == 0):
                counter0 += 1;
        compare_id.append(counter)
        compare_id0.append(counter0)
    class_comp = []
    for i in range(0,len(compare_id)):
        if compare_id[i] > compare_id0[i]:
            class_comp.append(1)
        if compare_id0[i] > compare_id[i]:
            class_comp.append(0)
    true_neg = 0;
    true_pos = 0;
    false_pos = 0;
    false_neg = 0;

    for i in range(len(test_class)):
        if (test_class[i]==1 and class_comp[i]==1):
            true_pos += 1;
        if (test_class[i]==1 and class_comp[i]==0):
            false_neg += 1;
        if (test_class[i]==0 and class_comp[i]==1):
            false_pos += 1;
        if (test_class[i]==0 and class_comp[i]==0):
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
    f1_score = (2 * (1/((1/precision)+(1/recall))));
    print("The accuracy in % is: ",accuracy)
    print("The precision in % is: ",precision)
    print("The recall in % is: ",recall)
    print("The F1 score in % is: ",f1_score)
    


if __name__ == "__main__":
    load_testdata();
    load_traindata();
    distance();
    nearest_neighbor();