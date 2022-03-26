#STEP 1: DATA CLEANING
from unicodedata import category
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
#Read in the Data
pricing = pd.read_csv('pricing.csv')
pricing.head()

#looking at plots to check for correlation between attributes
attributes = ["price","quantity","duration","category"]
pd.plotting.scatter_matrix(pricing[attributes], figsize=(12,8))#does this show up for anyone? Should be like 16 plots

#1. Check for nulls
pricing.isnull().any() #No nulls in any column 

#2. turn categorical values into dummies
pricing.dtypes #which columns are categorical?
#Doesn't look like any of them are...we should change that for sku & category based on the group assignment description of the data. 
pricing['category']=pricing['category'].astype('category')
pricing['sku']=pricing['sku'].astype('category')
pricing.dtypes #check that the categorical ones are now category

len(pricing.sku.unique())#total number unique values for sku is 74,999
len(pricing.category.unique()) #total number of unique values for category is 32
#now we can get dummies. Because sku and category are both category dtype, we don't have to define columns..it should automatically convert all columns with object or category dtype.
pricing_clean = pd.get_dummies(pricing)




