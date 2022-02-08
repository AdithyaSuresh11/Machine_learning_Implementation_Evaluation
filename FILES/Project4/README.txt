COURSE - CPSC 6430 - MACHINE LEARNING IMPLEMENTATION AND EVALUATION
PROFESSOR - DR. LARRY F HODGES

STUDENT - ADITHYA SURESH, C18590622

PROJECT 4:

FILES TO RUN:

1) Suresh_Adithya_P4.py for final results and all calculations.

DESCRIPTION:

1) The project requires a training datafile - (P4train.txt) and testing datafile
- (P4test.txt) to start with and when fed as input,it splits the columns and 
rows based on features and data (delimiter is a tab).

2) The data file was used to create additional features based on the logic insighted 
from Dr. Hodges. This occurs for both train and test files and the resultant new data
file is saved as Suresh_Adithya_P4train.txt and Suresh_Adithya_P4test.txt.

3) The data was then computed with initial alpha value of 0.5, number of iterations to be 
100,000 and weights were taken as 0 initially. After training the weights were calculated
along with the J value

4) Plot was plotted between J and no of iterations.

5) The weights from training was used to test the model and the J value for testing
was calculated.

6) The confusion matrix and accuracy, precision, recall and F1 score were calculated.

OUTPUT: IN THE ORDER:

Enter the name of your training data file: P4train.txt
Final Weights computed after training:  [[  3.90476209]
 [  1.65562789]
 [ -9.43385664]
 [  1.03847616]
 [  6.2798955 ]
 [-11.31262532]
 [ -7.97909416]
 [ 19.29342191]
 [ -0.7641728 ]
 [  2.60324433]
 [-12.44370207]
 [ 20.57528238]
 [  0.54824517]
 [ -6.30702865]
 [ -9.10553936]
 [  8.99854745]
 [-14.28637634]
 [-15.32711655]
 [-10.11198289]
 [  5.95742904]]
Total number of iterations involved:  100000
Final J after training is:  0.30481072071227416

Enter the name of your testing data file: P4test.txt
Final J after testing is 0.47262197486782836
True negative is:  14
True positive is:  13
False negative is:  4
False positive is:  2
The Confusion Matrix is:  [[14  2]
 [ 4 13]]
The accuracy in % is:  81.81818181818183
The precision in % is:  86.66666666666667
The recall in % is:  76.47058823529412
The F1 score in % is:  81.25000000000001