# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:48:56 2021

@author: adithya
"""

import numpy as np
import matplotlib.pyplot as plt

### For dataset name to be used in plot() function - use - P3train.txt and P3test.txt

def plot():
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
    
    file_test = input("Enter the name of your testing data file: ");
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
    # print(len(total_plot[:,2]))
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
        
def load_valdata():
    global val_data, val_class
    file = "validation_dataset/Val5.txt"
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
    
    val_data = np.zeros([row,column]);
    val_data = data[:,0:2];
    val_class = np.zeros([row,column-1])
    val_class = data[:,2:3] 
    # print(val_class)
    return val_data, val_class

def load_traindata():
    global train_data, train_class
    file = "train_dataset/Train5.txt"
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
    distance = np.zeros([len(val_data),len(train_data)]);
    for i in range(len(val_data)):
        val_pointx = val_data[a,b];
        val_pointy = val_data[a,b+1];
        a += 1;
        for j in range(len(train_data)):
            distance[i,j] = ((val_pointx - train_data[j,0])**2 + (val_pointy - train_data[j,1])**2)**0.5;  
    
    ids = np.argsort(distance)
    sort_distance = np.sort(distance)
    return ids, sort_distance

def nearest_neighbor():
    global k;
    # k = 3
    for k in range(1,25,2):
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
        missclassification = 0;
        for i in range(len(val_class)):
            if (val_class[i] != class_comp[i]):
                missclassification += 1;
        # print(missclassification)
    
#     return k, nearest_points, nearest_ids

def K_fold():
    file_train = [1,2,3,4,5];
    file_val = [1,2,3,4,5];
    for i in file_train:
        if i == 1:
            file_train = "train_dataset/Train1.txt"
            file_val = "validation_dataset/Val1.txt"
        if i == 2:
            file_train = "train_dataset/Train2.txt"
            file_val = "validation_dataset/Val2.txt"
        if i == 3:
            file_train = "train_dataset/Train3.txt"
            file_val = "validation_dataset/Val3.txt"
        if i == 4:
            file_train = "train_dataset/Train4.txt"
            file_val = "validation_dataset/Val4.txt"
        if i == 5:
            file_train = "train_dataset/Train5.txt"
            file_val = "validation_dataset/Val5.txt"
        file_train = open(file_train, "r"); # opens the mentioned file
    
        stringtoint_train = file_train.readline(); 
        splitlines_train = stringtoint_train.split("\t"); 
        row_train = int(splitlines_train[0]); # total number of data rows
        column_train = int(splitlines_train[1]); # total number of data columns except features
        data_train = np.zeros([row_train,column_train+1]); # array of ones to store the rows and columns from .txt file
        for k in range(row_train): # iterating through the rows
            aString_train = file_train.readline();
            t_train = aString_train.split("\t");
            for j in range(column_train+1): # iterating through columns including features
                data_train[k,j] = float(t_train[j]); # now data contains the plot data
    
        train_data = np.zeros([row_train,column_train]);
        train_data = data_train[:,0:2];
        train_class = np.zeros([row_train,column_train-1])
        train_class = data_train[:,2:3]

        
        file_val = open(file_val, "r"); # opens the mentioned file
    
        stringtoint_val = file_val.readline(); 
        splitlines_val = stringtoint_val.split("\t"); 
        row_val = int(splitlines_val[0]); # total number of data rows
        column_val = int(splitlines_val[1]); # total number of data columns except features
        data_val = np.zeros([row_val,column_val+1]); # array of ones to store the rows and columns from .txt file
        for k in range(row_val): # iterating through the rows
            aString_val = file_val.readline();
            t_val = aString_val.split("\t");
            for j in range(column_val+1): # iterating through columns including features
                data_val[k,j] = float(t_val[j]); # now data contains the plot data
        
        val_data = np.zeros([row_val,column_val]);
        val_data = data_val[:,0:2];
        val_class = np.zeros([row_val,column_val-1])
        val_class = data_val[:,2:3] 

        a = 0;
        b = 0;
        distance = np.zeros([len(val_data),len(train_data)]);
        for i in range(len(val_data)):
            val_pointx = val_data[a,b];
            val_pointy = val_data[a,b+1];
            a += 1;
            for j in range(len(train_data)):
                distance[i,j] = ((val_pointx - train_data[j,0])**2 + (val_pointy - train_data[j,1])**2)**0.5;  
        
        ids = np.argsort(distance)
        sort_distance = np.sort(distance)
        # print(sort_distance)
        # missclassification = np.zeros([len(ids),len(train_data)])
        # print(missclassification)
        for k in range(1,25,2):
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
            missclassification = 0;
            for i in range(len(val_class)):
                if (val_class[i] != class_comp[i]):
                    missclassification += 1;
                print(missclassification)
    # return ids, sort_distance

if __name__ == "__main__":
    plot();
    load_valdata();
    load_traindata();
    distance();
    nearest_neighbor();
    # K_fold();