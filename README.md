# CreditDefault
Credit Data Classification using k-Nearest Neighbors (k-NN)
This code demonstrates the implementation of a k-Nearest Neighbors (k-NN) classification model on a credit dataset. The goal is to predict whether a person will default on their credit based on their income, age, and loan information.

The code performs the following steps:

Imports the necessary libraries, including numpy, pandas, and modules from scikit-learn for data preprocessing, model training, and evaluation, as well as matplotlib.pyplot for visualizations.

Reads the credit dataset from a CSV file (credit_data.csv) using pandas and stores it in a DataFrame called creditData.

Selects the features 'income', 'age', and 'loan' from the dataset and assigns them to the variable features. The target variable 'default' is assigned to the variable target.

Reshapes the features into a 2D array using numpy to prepare the data for model training.

Converts the target variable to a numpy array.

Normalizes the features using MinMaxScaler from scikit-learn to scale the values between 0 and 1, ensuring each feature contributes equally to the model.

Splits the dataset into training and testing sets using train_test_split from scikit-learn, with a test size of 30% and the remaining 70% for training the model.

Creates a k-NN classification model with KNeighborsClassifier from scikit-learn and sets the number of neighbors (n_neighbors) to 32. The model is then trained on the training data using the fit method.

Makes predictions on the testing data using the trained model's predict method.

Performs cross-validation to determine the optimal value of k (number of neighbors). It iterates through values of k from 1 to 99, creates a new k-NN model for each k, and uses cross_val_score from scikit-learn to calculate the accuracy scores using 10-fold cross-validation. The mean accuracy score for each k is stored in a list.

Identifies the optimal value of k by finding the index of the maximum value in the list of mean accuracy scores using numpy.

Computes a confusion matrix using confusion_matrix from scikit-learn to evaluate the model's performance on the test data. It compares the actual target values (target_test) with the predicted values (predictions). The confusion matrix is displayed and plotted using ConfusionMatrixDisplay and matplotlib.pyplot.

![image](https://github.com/abhigyan02/CreditDefault/assets/75851981/7bdfe371-91f7-4c50-ba59-dc6cd5ac6d21)


Calculates the accuracy score by comparing the predicted values with the actual target values using accuracy_score from scikit-learn.

![Screenshot 2023-06-21 161453](https://github.com/abhigyan02/CreditDefault/assets/75851981/2ea48a46-4d85-4ff9-8add-4c7d8d83a9da)


Prints the optimal value of k determined through cross-validation.

Displays the accuracy score of the k-NN model on the test data.
