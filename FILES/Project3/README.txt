COURSE - CPSC 6430 - MACHINE LEARNING IMPLEMENTATION AND EVALUATION
PROFESSOR - DR. LARRY F HODGES

STUDENT - ADITHYA SURESH, C18590622

PROJECT 3:

FILES TO RUN:

1) Project3.py for fold creation

2) kNN.py for finding the misclassifications

3) Suresh_Adithya_P3.py for final results.

DESCRIPTION:

1) The project requires a training datafile - (P3train.txt) and testing datafile
- (P3test.txt) to start with and when fed as input,it splits the columns and 
rows based on features and data (delimiter is a tab).

2) Then an array of data is taken for both train and test dataset and the split into 5 folds
which calculates the euclidean distance between the validation point with respect to
the same training point. Then the k-NN was used to calculate the misclassification 

3) The misclassification for k values of five folds are taken and the total errors 
are pointed where the % accuracy was calculated. Then the k value was selected to be 5.

4) For the k value 5, test data and train data were used to calculate the euclidean 
distance based on which the confusion matrix with TP, TN, FP, FN was calculated.

5) The accuracy, precision, recall and F1 score was calculated on the values.

OUTPUT: (in the order):

Enter the name of your testing data file: P3train.txt

Enter the name of your training data file: P3test.txt
True negative is:  12
True positive is:  33
False negative is:  8
False positive is:  32
The Confusion Matrix is:  [[12 32]
 [ 8 33]]
The accuracy in % is:  52.94117647058824
The precision in % is:  50.76923076923077
The recall in % is:  80.48780487804879
The F1 score in % is:  62.26415094339623 