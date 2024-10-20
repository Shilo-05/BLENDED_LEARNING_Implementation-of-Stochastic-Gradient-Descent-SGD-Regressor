# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1. Import the necessary libraries
Import essential libraries like pandas, numpy, sklearn, and matplotlib for data handling, model creation, and visualization.

#### 2. Load the dataset
Use pandas.read_csv() to load the dataset into your environment for analysis.

#### 3. Preprocess the data
Handle any missing values by filling or removing them.
Convert categorical variables into numerical format using one-hot encoding or label encoding.

#### 4. Split the data into features (X) and target (y)
Separate the dataset into independent variables (features) and the dependent variable (target) for prediction.

#### 5. Divide the data into training and testing sets
Use train_test_split() to split the dataset into training and testing sets to build and evaluate the model.
#### 6. Create an SGD Regressor model
Initialize an SGDRegressor() model from sklearn.linear_model.


#### 7. Fit the model on the training data
Train the model using the .fit() method to learn the relationship between features and the target variable.


#### 8. Evaluate the model performance
Assess the model using metrics like Mean Squared Error (MSE) and R² score to check accuracy and performance.


#### 9. Make predictions and visualize the results
Use the trained model to make predictions on the test data and visualize the predicted vs actual values using plots like scatter plots.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by:  Oswald Shilo
RegisterNumber:  212223040139
*/
```

```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Data preprocessing
# Dropping unnecessary columns like 'CarName' and 'car_ID' since they don't contribute to predictions
data = data.drop(['CarName', 'car_ID'], axis=1)

# Handling categorical variables using one-hot encoding, dropping the first category to avoid multicollinearity
data = pd.get_dummies(data, drop_first=True)

# Define target variable (y) as 'price' and features (X) as the remaining columns
y = data['price']
X = data.drop(['price'], axis=1)

# Print the shape of features and target to check dimensions
print(X.shape, y.shape)

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the training and test sets to ensure proper splitting
print(X_train.shape, X_test.shape)

# Standardize the features to have mean = 0 and variance = 1
# This helps with optimizing the SGD algorithm, which is sensitive to feature scaling
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SGD Regressor model with 1000 iterations and a tolerance of 1e-3
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_regressor.fit(X_train_scaled, y_train)

# Predict on the test set using the trained model
y_pred = sgd_regressor.predict(X_test_scaled)

# Calculate and print the Mean Squared Error (MSE), which measures the average squared difference between actual and predicted values
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate and print the R-squared score, which explains the proportion of variance in the target variable explained by the model
r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2}")

# Plot the actual car prices vs the predicted car prices to visualize the model's performance
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/c6ebfc44-9a91-4972-ae2f-033905c07b44)



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
