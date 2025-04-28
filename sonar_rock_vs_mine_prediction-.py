"""
  https://colab.research.google.com/drive/1hxJfE1yqrBZrtHmuAhpyIeADu_ocnYi9
"""


#Importing the Dependencies

import numpy as np
import pandas as pd

#Data Collection & Processing

sonar_data = pd.read_csv('sonardata.csv.csv', header=None)

sonar_data.shape

sonar_data.describe()

sonar_data.std

sonar_data[60].value_counts()

# separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print(X)
print(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

#Model Evaluation

from sklearn.metrics import accuracy_score
#accuracy on training data
X_train_prediction = model.predict(X_train)    # X train pred
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f"Accuracy on training data is : {training_data_accuracy}")

#accuracy on test data
X_test_prediction = model.predict(X_test)    # X test pred
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Accuracy on test data is : {test_data_accuracy}")

"""Predictive System"""

while True:
    data_input = input("Enter the row number from the dataset (0 to 207) or 'stop' to exit: ")

    if data_input.lower() == 'stop':
        print("Exiting the program.")
        print("Thank you for using this program!😍")
        print("Goodbye!👋👋")
        break

    try:
        row_index = int(data_input)

        if row_index < 0 or row_index >= len(sonar_data):
            print(f"Error: Row number must be between 0 and {len(sonar_data)-1}. Please try again.")
            continue

        # Get the row as a DataFrame
        row_data = sonar_data.iloc[[row_index], :-1]
        print("\nSelected row data:")
        print(row_data)

        # Prepare data for prediction
        data = row_data.values.reshape(1, -1)

        # Make prediction
        prediction = model.predict(data)

        # Output prediction
        if prediction[0] == 'R':
            print('The object is a: Rock')
        else:
            print('The object is a: Mine')

    except ValueError:
        print("Error: Please enter a valid row number or 'stop'.")
    except Exception as e:
        print(f"An error occurred: {e}")