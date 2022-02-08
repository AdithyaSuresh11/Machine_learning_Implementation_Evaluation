# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:52:31 2021

@author: adithya
"""

import numpy as np
import matplotlib.pyplot as plt

def create_fold_files():
    filename = "P3train.txt";
    
    file = open(filename, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t");
    row = int(float(splitlines[0])); # total number of data rows
    column = int(float(splitlines[1])); # total number of data columns except features
    
    data = np.zeros([row,column+1]);
    for k in range(row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
    
        for j in range(column+1): # iterating through columns including features
            data[k,j] = float(t[j]); # now data contains the plot data
    
    empty = list();
    for i in range(1,(len(data)+1)):
        if (i%17==0):
            # print(i)
            empty.append(i);
    # print(empty)
    
    fold1 = data[0:(empty[0]),:];
    fold2 = data[(empty[0]):(empty[1]),:];
    fold3 = data[(empty[1]):(empty[2]),:];
    fold4 = data[(empty[2]):(empty[3]),:];
    fold5 = data[(empty[3]):(empty[4]),:];
    
    np.savetxt('fold_files/Fold1.txt', fold1, fmt="%0.5f",delimiter='\t',comments = "");
    np.savetxt('fold_files/Fold2.txt', fold2, fmt="%0.5f",delimiter='\t',comments = "");
    np.savetxt('fold_files/Fold3.txt', fold3, fmt="%0.5f",delimiter='\t',comments = "");
    np.savetxt('fold_files/Fold4.txt', fold4, fmt="%0.5f",delimiter='\t',comments = "");
    np.savetxt('fold_files/Fold5.txt', fold5, fmt="%0.5f",delimiter='\t',comments = "");

    with open('fold_files/Fold1.txt') as fp:
        fold1_data = fp.read();
    with open ('fold_files/Fold2.txt') as fp:
        fold2_data = fp.read();
    with open ('fold_files/Fold3.txt') as fp:
        fold3_data = fp.read();
    with open ('fold_files/Fold4.txt') as fp:
        fold4_data = fp.read();
    with open ('fold_files/Fold5.txt') as fp:
        fold5_data = fp.read();
        
    data1234 = fold1_data + fold2_data + fold3_data + fold4_data;
    data1235 = fold1_data + fold2_data + fold3_data + fold5_data;
    data1245 = fold1_data + fold2_data + fold4_data + fold5_data;
    data1345 = fold1_data + fold3_data + fold4_data + fold5_data;
    data2345 = fold2_data + fold3_data + fold4_data + fold5_data;
    
    header = '17\t2';
    train_header = '68\t2';
    with open ('train_dataset/Train5.txt','w') as fp:
        fp.write(train_header + "\n" + data1234);
    with open ('train_dataset/Train4.txt','w') as fp:
        fp.write(train_header + "\n" + data1235);
    with open ('train_dataset/Train3.txt','w') as fp:
        fp.write(train_header + "\n" + data1245);
    with open ('train_dataset/Train2.txt','w') as fp:
        fp.write(train_header + "\n" + data1345);
    with open ('train_dataset/Train1.txt','w') as fp:
        fp.write(train_header + "\n" + data2345);
        
    with open ('validation_dataset/Val5.txt','w') as fp:
        fp.write(header + "\n" + fold5_data);
    with open ('validation_dataset/Val4.txt','w') as fp:
        fp.write(header + "\n" + fold4_data);
    with open ('validation_dataset/Val3.txt','w') as fp:
        fp.write(header + "\n" + fold3_data);
    with open ('validation_dataset/Val2.txt','w') as fp:
        fp.write(header + "\n" + fold2_data);
    with open ('validation_dataset/Val1.txt','w') as fp:
        fp.write(header + "\n" + fold1_data);
       
if __name__ == "__main__":
    create_fold_files();
