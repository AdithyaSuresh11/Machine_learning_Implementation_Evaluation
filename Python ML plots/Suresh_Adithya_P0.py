# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 01:27:02 2021

@author: adithya
"""

####### COURSE: MACHINE LEARNING: IMPLEMENTATION AND EVALUATION ########
####### AUTHOR: ADITHYA SURESH, C18590622 ##########
####### PROJECT 0 ########

### For typing the dataset name, please use - IrisData.txt ###

### Numpy and Matplotlib libraries imported for use ###

import numpy as np
import matplotlib.pyplot as plt

### project function performs all expected tasks ###

def project():
    
    # opens the file based on input command, please use - IrisData.txt
    fileforopen = input("Enter the name of your data file: ");
    file = open(fileforopen, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t"); 
    usage = list(); # creating an empty list to store the data and features
    row = int(splitlines[0]); # total number of data rows
    column = int(splitlines[1]); # total number of data columns except features
    
    data = np.ones([row,column+1]); # array of ones to store the rows and columns from .txt file

    for k in range(row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
        for j in range(column+1): # iterating through columns including features
            if j==4:
                usage.append(t[j].strip()); # appending the data in empty list
                break;
            data[k,j] = float(t[j]); # now data contains the plot data
    
    print("""You can do a plot of any two features of the Iris Data set
            The feature codes are:
            0 = sepal length
            1 = sepal width
            2 = petal length
            3 = petal width""")
    
    # Requesting horizontal and vertical axes inputs
    hor_feature = input("Enter feature code for the horizontal axis: ");
    ver_feature = input("Enter feature code for the vertical axis: ");
    hor_int = int(hor_feature);
    ver_int = int(ver_feature);
    
    for line in range(row): 
        
        coloring = usage[line];
        colors = ['g','b','r']; # for the color coding of plot points
        
        if coloring == "setosa":
            l_sets = plt.scatter(float(data[line,hor_int]),float(data[line,ver_int]),color = colors[0], marker = "v");

        elif coloring == "versicolor":
            l_vers = plt.scatter(float(data[line,hor_int]),float(data[line,ver_int]),color = colors[1], marker = "o"); 

        elif coloring == "virginica":
            l_virg = plt.scatter(float(data[line,hor_int]),float(data[line,ver_int]),color = colors[2], marker = "+");    
    
    # Based on the user's inputs, the horizontal and vertical axes labels are printed
            
    if hor_int == 0:
        plt.xlabel("Sepal Length");
    elif hor_int == 1:
        plt.xlabel("Sepal Width");
    elif hor_int == 2:
        plt.xlabel("Petal Length");
    elif hor_int == 3:
        plt.xlabel("Petal Width");
    
    if ver_int == 0:
        plt.ylabel("Sepal Length");
    elif ver_int == 1:
        plt.ylabel("Sepal Width");
    elif ver_int == 2:
        plt.ylabel("Petal Length");
    elif ver_int == 3:
        plt.ylabel("Petal Width");
    
    # Location of the legend is fed as upper right as per the example plot
        
    plt.legend((l_sets, l_vers, l_virg),('Setosa','Versicolor','Virginica'), loc='upper right');
    plt.title("Iris Flower Plot");
    plt.savefig("result.png",bbox_inches="tight"); # result.png shows the latest run result
    plt.show();
    
    flag = 1;    
    while flag == 1:
        # Based on the string 'y' or 'n' - the program runs again or breaks
        # out of the while loop and terminates
        newplot = input("Would you like to do another plot? (y/n) ");
        if newplot == "y":
            if flag == 1:
                project();
                flag = 2;
        if newplot == "n":
            break;

if __name__ == "__main__":
    project(); # the project function is called in main function