""" This program explores linear regressions machine learning using the Boston Housing dataset. All linear regressions generated are trying to predict the 'MEDIAN VALUE,' 
    but the input used to make the prediction varries.  
"""
__version__ = '0.1'
__author__ = 'Amanda Foster'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Linear Regression with scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_reg = LinearRegression()

# Explore the Boston Housing data set
boston = pd.read_csv('boston_housing.csv')

"""
print(boston.info())
print(boston.head())

# Description of Boston Housing data set:
# CRIME RATE =  per capita crime rate by town
# LARGE LOT = proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUSTRY = proportion of non-retail business acres per town
# RIVER = Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX = nitric oxides concentration (parts per 10 million)
# ROOMS = average number of rooms per dwelling
# PRIOR 1940 = proportion of owner-occupied units built prior to 1940
# EMP DISTANCE = weighted distances to five Boston employment centres
# HWY ACCESS = index of accessibility to radial highways
# PROP TAX RATE = full-value property-tax rate per $10,000
# STU TEACH RATIO = pupil-teacher ratio by town
# AFR AMER = 1000(AFA - 0.63)^2 where AFA is the proportion of African Americans by town
# LOW STATUS = % lower status of the population
# MEDIAN VALUE = Median value of owner-occupied homes in $1000’s

# Creator: Harrison, D. and Rubinfeld, D.L.
# This is a copy of UCI ML housing dataset. https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

boston.hist(figsize=(14,7))
plt.title("Histogram")
plt.show()
boston.boxplot(figsize=(14,7))
plt.title("Box plot")
plt.show()

corr_matrix=boston.corr()
print(corr_matrix["MEDIAN VALUE"].sort_values(ascending=False))

pd.plotting.scatter_matrix(boston[ ['MEDIAN VALUE','LOW STATUS','ROOMS','INDUSTRY','NOX','PROP TAX RATE','STU TEACH RATIO'] ], figsize=(14,7))
plt.title("Scatter Matrix")
plt.show()

plt.scatter(boston['LOW STATUS'], boston['MEDIAN VALUE'])
plt.title("Scatter of Median Value(y) vs Low Status(x)")
plt.show()
"""

"""Setup a regression with a single variable
"""
def run_regression(input_value):
    # Setup a sample regression, using scikit
    boston_inputs = boston[ input_value ] # You can add more columns to this list...
    boston_targets = boston['MEDIAN VALUE']

    # Train the weights
    lin_reg.fit(boston_inputs,boston_targets)

    # Generate outputs / Make Predictions
    boston_outputs = lin_reg.predict(boston_inputs)

    # What's our error?
    boston_mse = mean_squared_error(boston_targets, boston_outputs)
    # What's our R^2? (amount of output variance explained by these inputs)
    boston_r2 = r2_score(boston_targets, boston_outputs)
    print("MSE using" + str(input_value) + " (scikit way): " + str(boston_mse*len(boston)))
    print("R^2 using" + str(input_value) + "(scikit way): " + str(boston_r2))
    print("Weights/Coefficients of Regression: " + str(lin_reg.coef_))

""" This method runs a linear regression to predict the mean value using the varaibles inputed 
""" 
def get_regression_error(input_value):
    boston_inputs = boston[ input_value ] # You can add more columns to this list...
    boston_targets = boston['MEDIAN VALUE']
     
    # Train the weights
    lin_reg.fit(boston_inputs,boston_targets)

    # Generate outputs / Make Predictions
    boston_outputs = lin_reg.predict(boston_inputs)

    # What's our error?
    boston_mse = mean_squared_error(boston_targets, boston_outputs)
    
    #returns the MSE for the regression
    return str(boston_mse*len(boston)) 

""" This method prints out the regression output of the variable in the inputed list that lowers the overall error by 3% or more
"""  
def find_lower_error(input_value):
    #gets orginal error without new input
    original_error = get_regression_error(['LOW STATUS SQUARED', 'LOW STATUS','ROOMS SQUARED', 'ROOMS'])
    
    # steps through all of the inputs
    for x in input_value:
        #gets the new error
        new_error = get_regression_error(['LOW STATUS SQUARED', 'LOW STATUS','ROOMS SQUARED', 'ROOMS', x])
        #prints out new error if its greater than a 3% improvement
        if float(original_error) * 0.97 > float(new_error): 
            run_regression(['LOW STATUS SQUARED', 'LOW STATUS','ROOMS SQUARED', 'ROOMS', x])

""" Main Method 
"""
if __name__ == "__main__":

    #1: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS as the input.
    print("Question 1")
    run_regression(['LOW STATUS'])
    
    #2: Predict MEDIAN VALUE in the Boston Housing dataset using ROOMS as the input.
    print("Question 2")
    run_regression(['ROOMS'])
    
    #3: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS AND Rooms as inputs.
    print("Question 3")
    run_regression(['ROOMS','LOW STATUS'])
   
    #4: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS and LOW STATUS^2 as inputs.
    print("Question 4")
    boston['LOW STATUS SQUARED'] = boston['LOW STATUS']**2
    run_regression(['LOW STATUS SQUARED', 'LOW STATUS'])
   
    #5: Predict MEDIAN VALUE in the Boston Housing dataset using ROOMS and ROOMS^2 as inputs.
    print("Question 5")
    boston['ROOMS SQUARED'] = boston['ROOMS']**2
    run_regression(['ROOMS SQUARED', 'ROOMS'])
   
    #6: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS, LOW STATUS^2, ROOMS, and ROOMS^2 as inputs.
    print("Question 6")
    run_regression(['LOW STATUS SQUARED', 'LOW STATUS','ROOMS SQUARED', 'ROOMS'])
    
    #7: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS, LOW STATUS^2, ROOMS, ROOMS^2, AND 'LOWROOMS' as inputs. LOWROOMS is an interaction term: LOW STATUS * ROOMS.
    print("Question 7")
    boston['LOWROOMS'] = boston['ROOMS'] * boston['LOW STATUS']
    run_regression(['LOW STATUS SQUARED', 'LOW STATUS','ROOMS SQUARED', 'ROOMS','LOWROOMS'])
    
    #8
    """ Starting with the inputs listed in #6, I created a list of all the other inputs. Using only one at a time I ran a new regression with these inputs and checked to see 
        if any of them have a greater than 3% improvement on the overall error. If this was the case I printed out the regression outputs. I found 4 inputs that had greater 
        than a 3% improvement: 'RIVER', 'PROP TAX RATE', 'STU TEACH RATIO', and 'AFR AMER'
        """
    print("Question 8")
    input_value = ['CRIME RATE','LARGE LOT','INDUSTRY', 'RIVER', 'NOX', 'PRIOR 1940', 'EMP DISTANCE', 'HWY ACCESS', 'PROP TAX RATE', 'STU TEACH RATIO', 'AFR AMER']
    find_lower_error(input_value)
    
    #9
    """ In order to mimimize the MSE I noticed that the more inputs you added the lower the MSE became. Meaning there is a negative correlation between MSE and the number of columns used
         as inputs. Because of this to get the lowest MSE I just added all the columns in the dataset and a few others that I made such as 'LOW STATUS SQUARED' and 'CRIME RATE SQUARED.'
         The lowest MSE value I attrained from my regression was 8348.702840309636
    """ 
    print("Question 9")
    boston['CRIME RATE SQUARED'] = boston['CRIME RATE']**2
    run_regression(['LOW STATUS SQUARED', 'LOW STATUS','ROOMS SQUARED', 'LOWROOMS', 'ROOMS', 'STU TEACH RATIO', 'PROP TAX RATE', 'RIVER','AFR AMER', 'LARGE LOT','CRIME RATE', 'INDUSTRY', 'RIVER', 'NOX', 'PRIOR 1940', 'HWY ACCESS', 'EMP DISTANCE', 'CRIME RATE SQUARED' ])
