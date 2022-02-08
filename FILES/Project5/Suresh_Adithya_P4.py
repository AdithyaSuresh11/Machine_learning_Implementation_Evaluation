# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:31:30 2021

@author: adithya
"""

import numpy as np
from matplotlib import pyplot as plt

def dataset(file):
    global data, row, column
    file = open(file, "r"); # opens the mentioned file
    
    stringtoint = file.readline(); 
    splitlines = stringtoint.split("\t"); 
    row = int(splitlines[0]); # total number of data rows
    column = int(splitlines[1]); # total number of data columns except features
    data = np.zeros([row,column]); # array of ones to store the rows and columns from .txt file

    for k in range(row): # iterating through the rows
        aString = file.readline();
        t = aString.split("\t");
        for j in range(column): # iterating through columns including features
            data[k,j] = float(t[j]); # now data contains the plot data
    return data,row,column

def centroid(centroid_file):
    global centroid_data, centroid_row, centroid_column
    centroid_file = open(centroid_file, "r"); # opens the mentioned file
    
    centroid_stringtoint = centroid_file.readline(); 
    centroid_splitlines = centroid_stringtoint.split("\t"); 
    centroid_row = int(centroid_splitlines[0]); # total number of data rows
    centroid_column = int(centroid_splitlines[0]); # total number of data columns except features
    centroid_data = np.zeros([centroid_row,centroid_column]); # array of ones to store the rows and columns from .txt file

    for k in range(centroid_row): # iterating through the rows
        centroid_aString = centroid_file.readline();
        centroid_t = centroid_aString.split("\t");
        for j in range(centroid_column): # iterating through columns including features
            centroid_data[k,j] = float(centroid_t[j]); # now data contains the plot data
    print("Initial Centroid1 is (",centroid_data[0][0],",",centroid_data[0][1],")")
    print("Initial Centroid2 is (",centroid_data[1][0],",",centroid_data[1][1],")")
    
    initial_centroid = np.array([[centroid_data[0][0],centroid_data[0][1]],[centroid_data[1][0],centroid_data[1][1]]]);
    print('Initial centroids are: ',initial_centroid);
    
    return centroid_data, centroid_row, centroid_column

def plot(data,centroid_data):
    empty = list();
    colors =["purple","red","green"]
    for line in range(row):
        data_points = plt.scatter(float(data[line,0]),float(data[line,1]),color = colors[0] , marker = "o");
    
    cent_1 = plt.scatter(float(centroid_data[0,0]),float(centroid_data[0,1]),color = colors[1] , marker = "v");
    cent_2 = plt.scatter(float(centroid_data[1,0]),float(centroid_data[1,1]),color = colors[2] , marker = "v");
    
    plt.title("Initial Data points");
    plt.xlabel("x1 Axis");
    plt.ylabel("x2 Axis");
    plt.legend((data_points, cent_1, cent_2),('Data points','Initial Centroid1','Initial Centroid2'), loc='upper right');
    plt.show();
    
    
def kmeans(data,centroid_data):
    flag = []
    colors =["red","green"]
    a = centroid_data[0][0];
    b = centroid_data[0][1];
    c = centroid_data[1][0];
    d = centroid_data[1][1];
    for loop in range(0,10):
        flag.clear();
        for i in range(0,len(data),1):
            x = data[i,0];
            y = data[i,1];
            dist1 = ((x - a)**2 + (y - b)**2)**(0.5);
            dist2 = ((x - c)**2 + (y - d)**2)**(0.5);
    
            if(dist1 < dist2):
                flag.append(1)
            elif (dist1 == dist2):
                pass;
            elif (dist1 > dist2):
                flag.append(2)
        a = b = c = d = 0;
        for i in range(0,len(flag)):
            if(flag[i] == 1):
                a+=(data[i,0]/flag.count(1))
                b+=(data[i,1]/flag.count(1))

            else:
                c+=(data[i,0]/flag.count(2))
                d+=(data[i,1]/flag.count(2))

    error = 0
    for i in range(0,len(flag)):
        if(flag[i] == 1):
            points = (data[i,0],data[i,1])
            clus1 = plt.scatter(points[0],points[1],color = colors[0] , marker = "o");
            center = (a,b)
            error+= ((points[0] - center[0])**2 + (points[1] - center[1])**2)
        
        else:
            points = (data[i,0],data[i,1])
            clus2 = plt.scatter(points[0],points[1],color = colors[1] , marker = "o");
            center = (c,d)
            error+= ((points[0] - center[0])**2 + (points[1] - center[1])**2)
    fin_cen1 = plt.scatter(a,b,color = colors[0] , marker = "v");
    fin_cen2 = plt.scatter(c,d,color = colors[1] , marker = "v");
    
    error/=len(data);
    
    plt.title("Final Data points");
    plt.xlabel("x1 Axis");
    plt.ylabel("x2 Axis");
    plt.legend((clus1, fin_cen1, clus2, fin_cen2),('Clustered points 1', 'Final Centroid1', 'Clustered points 2', 'Final Centroid2'), loc='upper right');
    plt.show();
    
    final_centroid = np.array([[a,b],[c,d]]);
    print('Final centroids are: ',final_centroid);
    print('Error is ',error);

if __name__ == "__main__":
    # file = "P5Data.txt";
    file = input("Enter the name of your data file: ");
    # centroid_file = "P5Centroids.txt";
    centroid_file = input("Enter the name of your centroid file: ");
    dataset(file);
    centroid(centroid_file);
    plot(data,centroid_data);
    kmeans(data,centroid_data);