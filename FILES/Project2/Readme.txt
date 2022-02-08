COURSE - CPSC 6430 - MACHINE LEARNING IMPLEMENTATION AND EVALUATION
PROFESSOR - DR. LARRY F HODGES

STUDENT - ADITHYA SURESH, C18590622

PROJECT 2:

1) The project requires a training datafile - (W100MTimes.txt) to start with and when fed as input,
it splits the columns and rows based on features and data (delimiter is a tab).

2) Then an array of ones is created where the x0, x1 and x2 (square of x1) values are inserted in for loop,
similarly the y values are taken from the file and added to the final column.

3) The weights are calculated using the formula, (X^T * X)^-1 * X^T *y 

4) Then the code prompts for the user to enter a year in the format - 2020 and based
on the input year the Women's winning race time for that year will be predicted.

Sample Output for the year 2005: [Output printed after the year prompt]:

Enter the year for winning women's prediction in format(eg. 2020): 2005
Predicted result of winning women's Olympic 100-meter race, for the year 2005 in seconds is  10.864869759926552