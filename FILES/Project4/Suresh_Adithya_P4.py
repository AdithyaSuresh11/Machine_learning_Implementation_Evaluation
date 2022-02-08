# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:31:30 2021

@author: adithya
"""

import numpy as np
from matplotlib import pyplot as plt

def kmeans():
    file = "P5Data.txt";
    # file = input("Enter the name of your training data file: ");
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


if __name__ == "__main__":
    kmeans();