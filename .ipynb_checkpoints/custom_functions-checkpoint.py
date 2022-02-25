import numpy as np
import pandas as pd
#create custom function to integrate into pipeline

#to replace nans with custome values
def nanreplace(data):
    #define replacement values and columns with null
    replace_value = ['Male', 'No', 0, 'No', 146.412162, 360, 1] 
    null_col = data.columns[data.isnull().any()]
    
    for i, col in enumerate(null_col):
            data[col] = data[col].fillna(replace_value[i])
            
    return data

#one to drop id col
def dropID(data):
    
    return data.drop(['Loan_ID'], axis=1)

#two for seperating out cat and int datatypes
def numFeat(data):
    #highlight the three numeric columns
    num_data = data[['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']]
    #coerce them back into numeric if they became objects
    num_data = num_data.apply(pd.to_numeric, errors='coerce')
    
    return num_data

def catFeat(data):
     #only return data with columns that have object dtype
    return data[data.columns[data.dtypes == 'object']]


def log_transform(data):
    #print(data)
    log_amount = np.log(data['LoanAmount'])
    log_income = np.log(data['ApplicantIncome'] + data['CoapplicantIncome'])

    #put into df and drop former columns
    data['LogAmount'] = log_amount
    data['LogIncomes'] = log_income
    
    return data[['LogAmount', 'LogIncomes']]
    

#categorical transformation step
def cat_transform(data):
    #fix Dependents
    data['Dependents'] = data['Dependents'].replace({'3+': 3})

    #convert into numeric
    data['Dependents'] = data['Dependents'].apply(pd.to_numeric, errors='coerce')
    
    #create list of columns to convert to dummies
    for_dummy = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

    #convert to dummy, append to original data and save as temp
    dummy = pd.get_dummies(data[for_dummy], drop_first = True)
    
    #in order to handle single row inputs we have to fuss with the data frame after
    #these are all the columns that should exist based on input data (fitted data)
    dummy_cols = ['Gender_Male', 'Married_Yes', 'Education_Not Graduate',
       'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban']
    
    #if we are missing any
    if len(dummy) < 6:
        #for every column that should be there
        for col in dummy_cols:
            #if its not there create column and set to 0 (false)
            if col not in dummy.columns:
                dummy[col] = 0
             
    #concatenate the dependents to dummy
    temp = pd.concat([data['Dependents'], dummy], axis =1)
    
    return temp