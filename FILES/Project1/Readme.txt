COURSE - CPSC 6430 - MACHINE LEARNING IMPLEMENTATION AND EVALUATION
PROFESSOR - DR. LARRY F HODGES

STUDENT - ADITHYA SURESH, C18590622

PROJECT 1:

1) The project requires a training datafile to start with and when fed as input,
it splits the columns and rows based on features and data (delimiter is a tab).

2) Then an array of ones is created where the x0 and x1 values are inserted in for loop,
similarly the y values are taken from the file and added to the final column.

3) The weights are calculated using the formula, (X^T * X)^-1 * X^T *y 
Then the training set J value is calculated using the formula, (1/m) * (X*w - y)^T * (X*w - y)

4) The code then prompts for the user to input the testing datafile and follows the above
procedure to calculate the J value for testing with the use of weights from training.

Output (in the order): [Run the code for reference]

Enter the name of your training data file: TrainCubed.txt
Computed weights during training are:  [[-1.28295759e-14]
 [-1.72414116e-11]
 [-6.73123245e-08]
 [ 4.23956540e-08]
 [-4.64272748e-10]
 [-1.64819933e-12]
 [ 2.06708869e-12]
 [-1.73371502e-08]
 [-2.98355610e-06]
 [-4.66176833e-06]
 [-8.50997935e-09]
 [-7.42656242e-11]
 [ 6.92028191e-10]
 [ 5.51477724e-09]
 [-1.15446828e-04]
 [ 6.93818747e-10]
 [-9.56746388e-08]
 [-2.68001748e-09]
 [ 1.37688515e-07]]
J value of the training dataset is  113.11572185607893

Enter the name of your testing data file: TestCubed.txt
J value of the testing dataset is  88.16835565641328