#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 19:13:17 2022
Last Updated on Fri Aug  4 11:02:24 2023
@author: moeezkhan
"""

import pandas as pd
import numpy as np
from scipy import stats                                      #to remove outliers using z-scores

from sklearn import model_selection                          #to get the cv score
from sklearn.linear_model import LogisticRegression          #to black box the model
from sklearn.linear_model import LinearRegression            #to black box the model

from sklearn.neighbors import KNeighborsRegressor            #to black box the model
from sklearn.ensemble import GradientBoostingRegressor       #to black box the model
from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt



# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
        
    demonstrateHelpers(trainDF)                                                #gives details about the dataframe

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperimentUsingLinearRegression(trainInput, trainOutput, predictors)     #implements the linearRegression model
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)

    

""" ===========================================================================
Models:
    1) KNNRegressor
    2) GradientBoostingRegressor
    3) Logistcalregression
    4) LinearRegression
    ~ models with parameter tuning
"""

'''
Model#1:
Does k-fold CV on the Kaggle training set using KNNRegressor.
'''
def doExperimentUsingKNNRegressor(trainInput, trainOutput, predictors, k):
    model = KNeighborsRegressor(n_neighbors = k)
    cvScores = model_selection.cross_val_score(model, trainInput, trainOutput, cv=k, scoring='r2')
    cvMeanScore = cvScores.mean()
    print("\nCV Average Score: ", cvMeanScore)
    
    
    
# =============================================================================
'''
Model#2:
Does k-fold CV on the Kaggle training set using GradientBoostingRegressor.
'''
def doExperimentUsingGradientBoostingRegressor(trainInput, trainOutput, predictors, k):
    # BEGIN: https://blog.paperspace.com/implementing-gradient-boosting-regression-python/
    # EXPLAINATION: The GBR algorhtim uses mutliple parameters (possible tuners) that help us implement the GBR algorithim
    model = GradientBoostingRegressor(n_estimators = 600,max_depth = 5,learning_rate = 0.01,min_samples_split=3)
    cvScores = model_selection.cross_val_score(model, trainInput.loc[:, predictors], trainOutput, cv=k, scoring='r2', n_jobs=-1)
    cvMeanScore = cvScores.mean()
    print("\nCV Average Score: ", cvMeanScore)
    # END: https://blog.paperspace.com/implementing-gradient-boosting-regression-python/
    


# =============================================================================
'''
Model#3:
Does k-fold CV on the Kaggle training set using LogisticalRegression.
'''
def doExperimentUsingLogisticalRegression(trainInput, trainOutput, predictors):
    model = LogisticRegression(solver = 'liblinear')
    cvScores = model_selection.cross_val_score(model, trainInput, trainOutput, cv = 10, scoring='r2')
    cvMeanScore = cvScores.mean()
    print("\nCV Average Score: ", cvMeanScore)



# =============================================================================
'''
Model#4:
Does k-fold CV on the Kaggle training set using LinearRegression.
'''
def doExperimentUsingLinearRegression(trainInput, trainOutput, predictors, k):
    model = LinearRegression()
    cvScores = model_selection.cross_val_score(model, trainInput.loc[:, predictors], trainOutput, cv=k, scoring='r2', n_jobs=-1)
    cvMeanScore = cvScores.mean()
    print("\nCV Average Score: ", cvMeanScore)


  
# =============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors, alg):
    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle



# =============================================================================
# Data cleaning - conversion, normalization
'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    predictors = ['1stFlrSF', '2ndFlrSF']
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    


# =============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')



# =============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues



# =============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues



# =============================================================================
'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)


# =============================================================================
'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)


# =============================================================================
def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs




"""  ==========================================================================
Pre-Processing
Steps:
    1) Converting categorical data into numeric data                [4 Methods]
    2) Scaling data using normalization and standardization         [2 Methods]
    3) Filling in the missing values                                [6 Methods]
    4) Handling outlier values                                      [1 Method]
    5) Finding the 'k' highly co-related 'SalePrice' attributes     [1 Method]
    6) remove duplicates
"""

#1)
#Converting Ordinal Attributes into Numeric Attributes
def convertOrdinal(trainDF):
    trainDF.loc[:, "LotShape"] = trainDF.loc[:, "LotShape"].map(lambda v: 1 if v =='Reg' else 2 if v == 'IR1' else 3 if v == 'IR2' else 4)
    trainDF.loc[:, "LandContour"] = trainDF.loc[:, "LandContour"].map(lambda v: 1 if v =='Lvl' else 2 if v == 'Bnk' else 3 if v == 'HLS' else 4)
    trainDF.loc[:, "LandSlope"] = trainDF.loc[:, "LandSlope"].map(lambda v: 1 if v =='Gtl' else 2 if v == 'Mod' else 3)
    trainDF.loc[:, "ExterQual"] = trainDF.loc[:, "ExterQual"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5)
    trainDF.loc[:, "ExterCond"] = trainDF.loc[:, "ExterCond"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5)

    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5 if v == "Po" else None)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5 if v == "Po" else None)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 1 if v =='Gd' else 2 if v == 'Av' else 3 if v == 'Mn' else 0 if v == "No" else None)
    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].map(lambda v: 1 if v =='GLQ' else 2 if v == 'ALQ' else 3 if v == 'BLQ' else 4 if v == 'Rec' else 5 if v == "LwQ" else 0 if v == 'Unf' else None)
    trainDF.loc[:, "BsmtFinType2"] = trainDF.loc[:, "BsmtFinType2"].map(lambda v: 1 if v =='GLQ' else 2 if v == 'ALQ' else 3 if v == 'BLQ' else 4 if v == 'Rec' else 5 if v == "LwQ" else 0 if v == 'Unf' else None)
    trainDF.loc[:, "HeatingQC"] = trainDF.loc[:, "HeatingQC"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5)
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5)

    trainDF.loc[:, "Functional"] = trainDF.loc[:, "Functional"].map(lambda v: 1 if v =='Typ' else 2 if v == 'Min1' else 3 if v == 'Min2' else 4 if v == 'Mod' else 5 if v == "Maj1" else 6 if v == 'Maj2' else 7 if v == 'Sev' else None)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5 if v == "Po" else None)
    trainDF.loc[:, "GarageFinish"] = trainDF.loc[:, "GarageFinish"].map(lambda v: 1 if v =='Fin' else 2 if v == 'RFn' else 3 if v == 'Unf' else None)
    trainDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5 if v == "Po" else None)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else 5 if v == "Po" else None)
    trainDF.loc[:, "PoolQC"] = trainDF.loc[:, "PoolQC"].map(lambda v: 1 if v =='Ex' else 2 if v == 'Gd' else 3 if v == 'TA' else 4 if v == 'Fa' else None)

    trainDF.loc[:, "Fence"] = trainDF.loc[:, "Fence"].map(lambda v: 1 if v =='GdPrv' else 2 if v == 'MnPrv' else 3 if v == 'GdWo' else 4 if v == 'MnWw' else None)

    

# =============================================================================
#Drop All the Nominal Attributes
def dropNominal(trainDF, ):
    nominalCols = ['MSZoning', 'Street', 'Alley', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                   'RoofMatl', 'Exterior1st','Exterior2nd', 'MasVnrType','Foundation', 'Heating', 'CentralAir', 'Electrical', 
                   'FireplaceQu', 'GarageType', 'PavedDrive', 'MiscFeature','SaleType', 'SaleCondition', 'SalePrice']
    trainDF = trainDF.drop(nominalCols, axis=1)
    return trainDF


# =============================================================================    
#Converting Binary Nominal Attributes into Numeric Attributes   
def convertNominalBinary(trainDF):
    trainDF.loc[:,"Street"].replace({'Grvl':0, 'Pave':1}, inplace = True)
    trainDF.loc[:,"CentralAir"].replace({'N':0, 'Y':1}, inplace = True)
        
    
    
# =============================================================================
#Converting Non-Binary Nominal Attributes into Numeric Attributes
def convertNominalMulticlass(trainDF):
    return pd.get_dummies(trainDF)



#2)
# =============================================================================
#Normalization on a Dataframe
def normalize(df, cols):
    df.loc[:,cols] = (df.loc[:,cols]- df.loc[:,cols].min()) / (df.loc[:,cols].max() - df.loc[:,cols].min())



# =============================================================================
#Standardization on a DataFrame
def standardize(df, cols):
    df.loc[:, cols] = (df.loc[:, cols] - df.loc[:,cols].mean()) / df.loc[:,cols].std()



#3)
# =============================================================================
#Find the Number of Missing Values For Each Attribute
def findMissingValueStats(trainDF, attributesWithMissingValues):
    missingValues = trainDF[attributesWithMissingValues].isnull().sum()[:15]
    sortedMissingValues = missingValues.sort_values(ascending=False)
    sortedMissingValues.plot(kind = 'bar', xlabel = 'Attributes', ylabel = 'Number Of Missing Values')
    plt.show()
    print(sortedMissingValues)
    
    

# =============================================================================
#Fill in the Missing Values With the Field's Mean Value
def fillMissingValuesWithMean(trainDF):  
    sourceDF = trainDF.copy()
    numAttributes = trainDF.shape[1]
    
    for attribute in range(numAttributes):
        trainDF.iloc[:, attribute] = trainDF.iloc[:, attribute].fillna(sourceDF.iloc[:, attribute].mean())
    
    """
    targetDF.loc[:, "LotFrontage"] = targetDF.loc[:, "LotFrontage"].fillna(sourceDF.loc[:, "LotFrontage"].mean())
    targetDF.loc[:, "Alley"] = targetDF.loc[:, "Alley"].fillna(sourceDF.loc[:, "Alley"].mean())
    targetDF.loc[:, "MasVnrType"] = targetDF.loc[:, "MasVnrType"].fillna(sourceDF.loc[:, "MasVnrType"].mean())
    targetDF.loc[:, "MasVnrArea"] = targetDF.loc[:, "MasVnrArea"].fillna(sourceDF.loc[:, "MasVnrArea"].mean())
    targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].fillna(sourceDF.loc[:, "BsmtQual"].mean())
    targetDF.loc[:, "BsmtCond"] = targetDF.loc[:, "BsmtCond"].fillna(sourceDF.loc[:, "BsmtCond"].mean())
    targetDF.loc[:, "BsmtExposure"] = targetDF.loc[:, "BsmtExposure"].fillna(sourceDF.loc[:, "BsmtExposure"].mean())
    targetDF.loc[:, "BsmtFinType1"] = targetDF.loc[:, "BsmtFinType1"].fillna(sourceDF.loc[:, "BsmtFinType1"].mean())
    targetDF.loc[:, "BsmtFinType2"] = targetDF.loc[:, "BsmtFinType2"].fillna(sourceDF.loc[:, "BsmtFinType2"].mean())
    targetDF.loc[:, "Electrical"] = targetDF.loc[:, "Electrical"].fillna(sourceDF.loc[:, "Electrical"].mean())
    targetDF.loc[:, "FireplaceQu"] = targetDF.loc[:, "FireplaceQu"].fillna(sourceDF.loc[:, "FireplaceQu"].mean())
    targetDF.loc[:, "GarageType"] = targetDF.loc[:, "GarageType"].fillna(sourceDF.loc[:, "GarageType"].mean())
    targetDF.loc[:, "GarageYrBlt"] = targetDF.loc[:, "GarageYrBlt"].fillna(sourceDF.loc[:, "GarageYrBlt"].mean())
    targetDF.loc[:, "GarageFinish"] = targetDF.loc[:, "GarageFinish"].fillna(sourceDF.loc[:, "GarageFinish"].mean())
    targetDF.loc[:, "GarageQual"] = targetDF.loc[:, "GarageQual"].fillna(sourceDF.loc[:, "GarageQual"].mean())
    targetDF.loc[:, "GarageCond"] = targetDF.loc[:, "GarageCond"].fillna(sourceDF.loc[:, "GarageCond"].mean())
    targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].fillna(sourceDF.loc[:, "PoolQC"].mean())
    targetDF.loc[:, "Fence"] = targetDF.loc[:, "Fence"].fillna(sourceDF.loc[:, "Fence"].mean())
    targetDF.loc[:, "MiscFeature"] = targetDF.loc[:, "MiscFeature"].fillna(sourceDF.loc[:, "MiscFeature"].mean())
    """


# =============================================================================
#Fill in the Missing Values With the Field's Mode Value
def fillMissingValuesWithMode(trainDF):  
    sourceDF = trainDF.copy()
    numAttributes = trainDF.shape[1]
    
    for attribute in range(numAttributes):
        trainDF.iloc[:, attribute] = trainDF.iloc[:, attribute].fillna(sourceDF.iloc[:, attribute].mode()[0])
    
    """
    targetDF.loc[:, "LotFrontage"] = targetDF.loc[:, "LotFrontage"].fillna(sourceDF.loc[:, "LotFrontage"].mode()[0])
    targetDF.loc[:, "Alley"] = targetDF.loc[:, "Alley"].fillna(sourceDF.loc[:, "Alley"].mode()[0])
    targetDF.loc[:, "MasVnrType"] = targetDF.loc[:, "MasVnrType"].fillna(sourceDF.loc[:, "MasVnrType"].mode()[0])
    targetDF.loc[:, "MasVnrArea"] = targetDF.loc[:, "MasVnrArea"].fillna(sourceDF.loc[:, "MasVnrArea"].mode()[0])
    targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].fillna(sourceDF.loc[:, "BsmtQual"].mode()[0])
    targetDF.loc[:, "BsmtCond"] = targetDF.loc[:, "BsmtCond"].fillna(sourceDF.loc[:, "BsmtCond"].mode()[0])
    targetDF.loc[:, "BsmtExposure"] = targetDF.loc[:, "BsmtExposure"].fillna(sourceDF.loc[:, "BsmtExposure"].mode()[0])
    targetDF.loc[:, "BsmtFinType1"] = targetDF.loc[:, "BsmtFinType1"].fillna(sourceDF.loc[:, "BsmtFinType1"].mode()[0])
    targetDF.loc[:, "BsmtFinType2"] = targetDF.loc[:, "BsmtFinType2"].fillna(sourceDF.loc[:, "BsmtFinType2"].mode()[0])
    targetDF.loc[:, "Electrical"] = targetDF.loc[:, "Electrical"].fillna(sourceDF.loc[:, "Electrical"].mode()[0])
    targetDF.loc[:, "FireplaceQu"] = targetDF.loc[:, "FireplaceQu"].fillna(sourceDF.loc[:, "FireplaceQu"].mode()[0])
    targetDF.loc[:, "GarageType"] = targetDF.loc[:, "GarageType"].fillna(sourceDF.loc[:, "GarageType"].mode()[0])
    targetDF.loc[:, "GarageYrBlt"] = targetDF.loc[:, "GarageYrBlt"].fillna(sourceDF.loc[:, "GarageYrBlt"].mode()[0])
    targetDF.loc[:, "GarageFinish"] = targetDF.loc[:, "GarageFinish"].fillna(sourceDF.loc[:, "GarageFinish"].mode()[0])
    targetDF.loc[:, "GarageQual"] = targetDF.loc[:, "GarageQual"].fillna(sourceDF.loc[:, "GarageQual"].mode()[0])
    targetDF.loc[:, "GarageCond"] = targetDF.loc[:, "GarageCond"].fillna(sourceDF.loc[:, "GarageCond"].mode()[0])
    targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].fillna(sourceDF.loc[:, "PoolQC"].mode()[0])
    targetDF.loc[:, "Fence"] = targetDF.loc[:, "Fence"].fillna(sourceDF.loc[:, "Fence"].mode()[0])
    targetDF.loc[:, "MiscFeature"] = targetDF.loc[:, "MiscFeature"].fillna(sourceDF.loc[:, "MiscFeature"].mode()[0])
    """


# =============================================================================
#Fill in the Missing Values With the Field's Median Value
def fillMissingValuesWithMedian(trainDF):  
    sourceDF = trainDF.copy()
    numAttributes = trainDF.shape[1]
    
    for attribute in range(numAttributes):
        trainDF.iloc[:, attribute] = trainDF.iloc[:, attribute].fillna(sourceDF.iloc[:, attribute].median())
        
    """
    targetDF.loc[:, "LotFrontage"] = targetDF.loc[:, "LotFrontage"].fillna(sourceDF.loc[:, "LotFrontage"].median())
    targetDF.loc[:, "Alley"] = targetDF.loc[:, "Alley"].fillna(sourceDF.loc[:, "Alley"].median())
    targetDF.loc[:, "MasVnrType"] = targetDF.loc[:, "MasVnrType"].fillna(sourceDF.loc[:, "MasVnrType"].median())
    targetDF.loc[:, "MasVnrArea"] = targetDF.loc[:, "MasVnrArea"].fillna(sourceDF.loc[:, "MasVnrArea"].median())
    targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].fillna(sourceDF.loc[:, "BsmtQual"].median())
    targetDF.loc[:, "BsmtCond"] = targetDF.loc[:, "BsmtCond"].fillna(sourceDF.loc[:, "BsmtCond"].median())
    targetDF.loc[:, "BsmtExposure"] = targetDF.loc[:, "BsmtExposure"].fillna(sourceDF.loc[:, "BsmtExposure"].median())
    targetDF.loc[:, "BsmtFinType1"] = targetDF.loc[:, "BsmtFinType1"].fillna(sourceDF.loc[:, "BsmtFinType1"].median())
    targetDF.loc[:, "BsmtFinType2"] = targetDF.loc[:, "BsmtFinType2"].fillna(sourceDF.loc[:, "BsmtFinType2"].median())
    targetDF.loc[:, "Electrical"] = targetDF.loc[:, "Electrical"].fillna(sourceDF.loc[:, "Electrical"].median())
    targetDF.loc[:, "FireplaceQu"] = targetDF.loc[:, "FireplaceQu"].fillna(sourceDF.loc[:, "FireplaceQu"].median())
    targetDF.loc[:, "GarageType"] = targetDF.loc[:, "GarageType"].fillna(sourceDF.loc[:, "GarageType"].median())
    targetDF.loc[:, "GarageYrBlt"] = targetDF.loc[:, "GarageYrBlt"].fillna(sourceDF.loc[:, "GarageYrBlt"].median())
    targetDF.loc[:, "GarageFinish"] = targetDF.loc[:, "GarageFinish"].fillna(sourceDF.loc[:, "GarageFinish"].median())
    targetDF.loc[:, "GarageQual"] = targetDF.loc[:, "GarageQual"].fillna(sourceDF.loc[:, "GarageQual"].median())
    targetDF.loc[:, "GarageCond"] = targetDF.loc[:, "GarageCond"].fillna(sourceDF.loc[:, "GarageCond"].median())
    targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].fillna(sourceDF.loc[:, "PoolQC"].median())
    targetDF.loc[:, "Fence"] = targetDF.loc[:, "Fence"].fillna(sourceDF.loc[:, "Fence"].median())
    targetDF.loc[:, "MiscFeature"] = targetDF.loc[:, "MiscFeature"].fillna(sourceDF.loc[:, "MiscFeature"].median())
    """
    
    
   
# =============================================================================
#Fill in the Missing Values With The 'K' Nearest Neighbors' Mean Value 
def fillMissingValuesWithKNNImputer(trainDF, k):
    # BEGIN: from https://betterdatascience.com/impute-missing-data-with-python-and-knn/
    # EXPLANATION: Imputers can fill in missing values with the mean by default.We are using KNN Imputer 
    # which calculate mean of k neighbors and assign it to missing blocks. # Filling in missing values is usually better than 
    # dropping them entirely because those that are not missing might indicate some valuable patterns.
    # so [:-1] means to slice up to but not including last char for c in s[:-1]: print(c, end='-') print(s[-1])
    imputer = KNNImputer(n_neighbors=k)
    imputed = imputer.fit_transform(trainDF)
    df_imputed = pd.DataFrame(imputed, columns=trainDF.columns)
    trainDF = df_imputed
    # END: from https://betterdatascience.com/impute-missing-data-with-python-and-knn/

    return trainDF
        
    
# =============================================================================
#Drop All Attributes That Have Missing Values
def dropMissingValues(trainDF, missingAttributes):
    trainDF = trainDF.drop(missingAttributes, axis=1)
    return trainDF


#4)
# =============================================================================
#Drop All Rows That Have Outlier Values For Any Attributes
def dropOutlierRows(trainDF):
    trainDF[(np.abs(stats.zscore(trainDF)) < 3).all(axis=1)]
    
    
#5)
# =============================================================================
#Find the Attributes that Are Highly Co-Related With the "SalePrice" and Treat Them As Predctors
def kHighestCorelatedAttributes(trainDF, k):
    correlations = trainDF.corr()['SalePrice']
    sortedCorrelations = correlations.sort_values(ascending = False)
    sortedCorrelations[1 : k + 1].plot(kind = 'area', xlabel = 'Attributes', ylabel = 'SalePrice Co-Relation')
    plt.show()
    
    vsGrLivArea = pd.concat([trainDF.loc[:, 'SalePrice'], trainDF.loc[:, 'OverallQual']], axis=1)
    vsGrLivArea.plot.scatter(x='OverallQual', y='SalePrice', figsize=(4, 6))
    
    vsLotArea = pd.concat([trainDF.loc[:, 'SalePrice'], trainDF.loc[:, '1stFlrSF']], axis=1)
    vsLotArea.plot.scatter(x='1stFlrSF', y='SalePrice', figsize=(4, 6))    
    plt.show()
    
    return sortedCorrelations[1 : k + 1]
    
    
    
"""Testing"""
"""
Steps:
1) Get And Prepare Data
2) Conversion
3) Missing Values
4) Scaling
5) Run Models
"""
# =============================================================================
#if __name__ == "__main__": 
def test():
    #main()
    
    """Get And Prepare Data"""
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    trainOutput = trainDF.loc[:, 'SalePrice']
    predictors = list(kHighestCorelatedAttributes(trainDF, 10).index)        #get the 'k' highest co-related attributes
    fullTrainDF = trainDF.copy()
    trainDF = trainDF.drop(columns = ['SalePrice'])
    
    
    """Convert Data"""
    convertOrdinal(trainDF)                                     #convert all ordinal data types into numeric
    convertNominalBinary(trainDF)                               #convert all nominal binary attributes into numeric
    trainDF = convertNominalMulticlass(trainDF)                 #convert all other nominal attributes into numeric
    #droppedTrainDF = dropNominal(trainDF)                      #drop the nominal values - use if nominal not yet numeric
    
    
    """Deal With Missing Values"""
    missingAttributes = getAttrsWithMissingValues(trainDF)      #get all the attributes with missing values
    #findMissingValueStats(trainDF, missingAttributes)    
    #trainDF = dropMissingValues(trainDF, missingAttributes)
    #fillMissingValuesWithMean(trainDF)                          #fill missing values with
    #fillMissingValuesWithMode(trainDF)                          #fill missing values with 
    #fillMissingValuesWithMedian(trainDF)
    trainDF = fillMissingValuesWithKNNImputer(trainDF, 10)         #fill missing values with
    
    #dropOutlierRows(trainDF)

    
    """Scaling Data"""
    validAttributes = trainDF.iloc[0, :].index                  #get all names of numeric attributes
    #normalize(trainDF, validAttributes)                         #normalize the training set                                             
    #standardize(trainDF, validAttributes)                       #standardize the trainings set
    
    """"Visualization"""
    cvScores = []
    modelNames = ['LinearRegression', 'kNNRegressor', 'GBR']
    
    
    """Use Models To Predict "SalePrice"""
    #cvScores.append(doExperimentUsingLinearRegression(trainDF, trainOutput, predictors, 8))
    #cvScores.append(doExperimentUsingKNNRegressor(trainDF, trainOutput, predictors, 8))
    #cvScores.append(doExperimentUsingGradientBoostingRegressor(trainDF, trainOutput, predictors, 8))
    
    #print(trainDF.iloc[:, :])
    
    trainInput, testInput, trainOutput, testIDs, predictors = transformData(fullTrainDF, testDF)    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors, GradientBoostingRegressor(n_estimators=600,max_depth=5,learning_rate=0.01,min_samples_split=3))


test()




# Another approach (not being used)
# =============================================================================
# Ordinal --> Numeric Conversion Maps
'''
nonNumericAttributeMap = getAttrToValuesDictionary(trainDF)
ordinalAttributes = {"LotShape","LandContour", "LandSlope", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                 "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond",
                 "GarageCond", "Fence", "SaleCondition"}

print("Here")
print(nonNumericAttributeMap["ExterQual"])

for attribute in nonNumericAttributeMap:
    if attribute in ordinalAttributes:
        attributeValues = nonNumericAttributeMap[attribute]
        count = 1
        for value in attributeValues:
            trainDF.loc[:, attribute] = trainDF.loc[:, attribute].map(lambda element : count + 1 if element == value else element)
'''
