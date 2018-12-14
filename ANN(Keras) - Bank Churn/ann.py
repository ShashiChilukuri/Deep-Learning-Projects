# Artificial Neural Network (ANN)#
##################################

#-----------------------------
# Part 1 - Data Preprocessing
#-----------------------------
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data - Country, Gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - To make sure all the features have same importance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#----------------------------
# Part 2 - Creating ANN model
#----------------------------

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # To initiate ANN model
from keras.layers import Dense # To create layers to ANN model

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
    # For this we have 6 hidden layers (based on avg of input and output variables)
    # For initializing we will use 'Uniform distribution' function
    # For ativation we will use rectifier
    # We got 11 inputs
    # We can use K-fold cross-validation to choose right layers based parameter tuning
    # Activation function that we will use is rectifier action for hidden layer, since closer the activation function value is to 1, the more activated is the neuron
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
    # Activation function that we will use is sigmoid action for layer since we are tying to predict the outcome
    # For binary outcome we will use Sigmoid function and for 3 or more categories we will use softmax activation function
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
    # Optimizer algorithm is used to choose optimal set of weights.
    # For Stochastic Gradient Descent, we got lot of optimizer algorithms. we will use adam for our model
    # For binnary outcome, Cost/loss function that we use in the optimizer is  'binary_crossentropy'
    # For 3 or more categorical outcome, Cost/loss function that we use in the optimizer is  'category_crossentropy'
    # Metrics is the cretarian that we choose to evaluate the model. one we use is 'accuracy'
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
    # we need to choose optimam parameter by experiment
    # we will input our traing set
    # We can pass on weights either after each observation or after batch of observations
    # nb_epoch is number of time we are training out model on whole traing set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# out training set got accuracy of 86%. now lets test this on our test set
#---------------------------------------------------------
# Part 3 - Making the predictions and evaluating the model
#---------------------------------------------------------
# Predicting the Test set results
    # predicted outcome are in probability.
    # We need to make the outcome to binary to compare with the actual outcome
    # We will consider 50% Theshold to convert the probabilites into true or false
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Conclusion
    # When tested this model on our test set, we got an accuracy of 86%.
    # To conclude, we trained and tested a ANN model for bank data to predicat that whether a customer will leave the back with 86% accuray.
    # Bank can use this model to test on all the customers, rank and categorize them based on probablity to leave the back
    # Then can focus on customer segment with high probability on leaving and take necessary measures to stop loosing them
